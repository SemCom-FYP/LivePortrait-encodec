# Neural web conferencing

A custom Python conferencing application built on **aiortc**, where neither
audio nor video uses a conventional codec:

| stream | what actually travels | measured rate |
|--------|----------------------|---------------|
| video  | LivePortrait implicit-keypoint motion vectors, 150 B/frame | **~17 kbps** at 14 fps, 24 kbps at 20 fps |
| audio  | EnCodec 24 kHz residual-VQ codes, bit-packed at 10 bits/entry | **~6.2 kbps** |

For comparison, H.264 at 640×480/25 fps in a conferencing configuration is
typically budgeted around 460 kbps. The whole call — both directions of one
stream, sound included — fits in roughly 5% of that.

The trick is that pixels are never transmitted. Each participant sends a
portrait of themselves **once** at call setup (~67 KiB of JPEG), and from then
on sends only what their face is *doing*. Every receiver re-synthesises them
locally with LivePortrait's warping and generator networks.

## Install

The repo's own LivePortrait requirements must already be working (models
downloaded into `pretrained_weights/`). Then:

```bash
pip install -r webrtc_conference/requirements.txt
```

## Run

Start the signaling server once, anywhere both participants can reach:

```bash
python -m webrtc_conference.signaling --port 8765
```

Then each participant runs a client:

```bash
python -m webrtc_conference.client \
    --room demo --name Alice \
    --avatar assets/examples/source/s6.jpg \
    --signaling ws://<server-host>:8765/ws
```

```bash
python -m webrtc_conference.client \
    --room demo --name Bob \
    --avatar assets/examples/source/s9.jpg \
    --signaling ws://<server-host>:8765/ws
```

A grid window opens with your own camera feed plus one re-synthesised tile per
remote participant, each labelled with its live bitrate.

Both clients on one machine can only share the webcam if they are told to use
different ones — `cv2.VideoCapture` fails on the second, so pass `--camera 1`
to the second client, or `--no-audio` and a second camera index when you just
want to watch the mesh come up.

| key | action |
|-----|--------|
| `q` / `Esc` | leave the call |
| `r` | re-calibrate your neutral pose (do this while facing the camera straight on) |
| `s` | save a PNG of the current grid |

Useful flags: `--fps` (motion vectors per second), `--bandwidth`
(EnCodec kbps: 1.5/3/6/12/24), `--audio-chunk-ms` (latency vs. overhead),
`--max-render-fps` (cap GPU spend per peer), `--preview-self`,
`--no-audio`, `--list-devices`, `--cpu`.

## Verify without a second machine

```bash
python -m webrtc_conference.selftest
```

Five stages: wire format round-trips, EnCodec encode→decode, LivePortrait
extract→render (writes `selftest_out/selftest_video.png` showing
*source | driving crop | reconstruction*), two real aiortc peers over loopback,
and two clients meeting through the actual signaling server.

## How it works

### Transport

No RTP media sections are negotiated at all. Everything rides four out-of-band
(pre-negotiated) SCTP DataChannels with fixed stream ids, so both ends agree on
the layout with no DCEP round-trip:

| id | label | reliability | carries |
|----|-------|-------------|---------|
| 1 | `ctrl` | reliable, ordered | JSON hello/bye |
| 2 | `blob` | reliable, ordered | the avatar JPEG, chunked |
| 3 | `motion` | **unreliable, unordered** (`maxRetransmits=0`) | motion vectors |
| 4 | `audio` | **time-bounded** (`maxPacketLifeTime=300ms`) | EnCodec chunks |

Motion and audio are deliberately lossy. Retransmitting a 50 ms-old facial
expression costs more than skipping it, and the motion channel additionally
drops frames rather than queueing whenever `bufferedAmount` climbs.

### The video "codec"

Per frame the sender runs only the motion extractor **M** on the cropped face
and ships 70 numbers:

```
pitch, yaw, roll        3    int16, 1/128 deg
scale                   1    int16, 1/8192
translation             3    int16, 1/8192
expression deltas    21×3    int16, 1/16384
                     ----
                       70 × 2 B  + 10 B header  = 150 B
```

Fixed-point rather than float16: the expression deltas live around 1e-2, where
float16's relative mantissa is coarser than a flat 6e-5 step. Round-trip error
is under 1e-4 (checked in stage 1 of the self-test).

The receiver already paid the expensive one-time costs when the avatar arrived —
the appearance volume `f_s` from **F** and the transformed source keypoints
`x_s` (~2 s of setup). Each arriving vector is composed against those with the
same relative-motion equation the offline pipeline uses, then warped and decoded
by **W**+**G**.

The neutral reference pose is carried in-band: the first packet of a stream sets
it, and pressing `r` sets the `FLAG_CALIBRATE` bit to reset it. Sender and
receiver therefore never need to agree on anything beyond the packets.

### The audio codec

EnCodec's 24 kHz model has no streaming framing, so it is driven in fixed chunks
(240 ms by default). Encoding chunks independently would leave a seam at every
boundary, so the encoder carries an 80 ms lookback into the model and discards
the corresponding latent frames — the convolution stack sees real context at the
boundary instead of zero padding — and the decoder crossfades 5 ms across joins.

Codes go out as a dense 10-bit bitstream rather than one `uint16` per entry,
which is what keeps the wire rate (6.23 kbps) at the model's nominal 6.0.

### Threading

aiortc owns an asyncio loop on a background thread; OpenCV owns the main thread.
Torch never runs on the loop. Capture, motion extraction, rendering and audio
coding each have their own thread and hand finished buffers across with
`call_soon_threadsafe` or a one-deep mailbox that drops stale frames instead of
queueing them.

## Performance and limits

Measured on an RTX 4060 Laptop GPU:

| step | cost | runs |
|------|------|------|
| motion extraction (M + crop) | ~25 ms | once per frame you send |
| warp + decode (W + G) | **~97 ms** | **once per remote participant per frame** |
| EnCodec encode / decode | ~17 ms / ~17 ms | per 240 ms chunk, on CPU |
| avatar setup (F, one-time) | ~2 s | once per peer joining |

Rendering is the ceiling, and it is the one cost that scales with room size:
about **10 fps with one remote peer, ~3 fps each with three**. Peers take strict
turns on the GPU and each renders only its newest vector, so latency stays flat
as the room grows — the frame rate degrades instead of a backlog forming.
`--max-render-fps` caps the spend explicitly.

Things worth knowing before relying on this:

- **Bitrate is not the bottleneck; the GPU is.** The obvious next optimisation
  is batching warp+decode across peers into one forward pass, which would turn
  N×97 ms into something much closer to 97 ms.
- **Audio latency** is roughly one chunk plus coding, ~300 ms one way at the
  default. `--audio-chunk-ms 120` halves it at some bitrate overhead.
- **Mesh topology.** Every peer connects to every other, so this is sensible up
  to about four participants. Beyond that it wants an SFU — though note the
  uplink cost of forwarding 150-byte frames is trivial.
- **You are rendered as your avatar, not as yourself.** Fidelity is bounded by
  how well the portrait matches you, and lighting or occlusions in the driving
  video do not transfer. Anyone holding your portrait can also re-render it from
  captured motion vectors.
- **`KMP_DUPLICATE_LIB_OK`** is set in `__init__.py`: onnxruntime and torch each
  ship an Intel OpenMP runtime and on Windows the second to load aborts the
  process.
- **No TURN by default.** Only a STUN server is configured, which is enough for
  most home NATs. A relay is needed when the two peers cannot reach each other
  directly at all — symmetric NAT, and, more commonly, **wifi with client
  isolation**, which most campus and guest networks enable. There neither host
  nor server-reflexive candidates are usable and the call only connects through
  TURN:

  ```bash
  python -m webrtc_conference.client ...       --turn turn:relay.example.org:3478       --turn-user USER --turn-pass SECRET
  ```

  Credentials are required: TURN URLs alone will not authenticate. `TURN_USER`
  and `TURN_PASS` work instead of the flags. To test whether your network needs
  this, see if the two machines can reach each other directly at all — serve the
  signaling server on the LAN and open `http://<host>:8765/rooms` from the other
  machine. If that fails, client isolation is on and TURN is mandatory.
