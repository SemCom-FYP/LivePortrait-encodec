# Offline run of the conference codec

`run_offline.py` pushes a video file (and its audio) through the same sender,
wire format and receiver classes that `webrtc_conference/client.py` uses in a
live call, and writes the video the far side would see. No webcam, mic,
WebRTC or UI is involved, so it runs on the GPU server.

```
video frames @ --fps ─► MotionEncoder.extract ─► pack_motion (150 B) ─┐
                                                                     │ bytes
audio, 240 ms chunks ─► EnCodec 6 kbit/s ─► pack_audio ──────────────┤ (optional
portrait (once) ─► JPEG ─────────────────────────────────────────────┤  motion loss)
                                                                     ▼
          AvatarRenderer.render (W+G, 512x512)  +  EnCodec decode ─► output.mp4
```

## Setup (once)

```bash
conda activate video_semantics_receiver
python -m pip install --no-deps encodec einops
# reuse the LivePortrait weights already on the server
ln -s ~/FYP_SemCon/video_semantics_receiver/pretrained_weights/liveportrait ~/FYP_SemCon/LivePortrait-encodec/pretrained_weights/liveportrait
ln -s ~/FYP_SemCon/video_semantics_receiver/pretrained_weights/insightface  ~/FYP_SemCon/LivePortrait-encodec/pretrained_weights/insightface
```

EnCodec downloads its 24 kHz weights (~90 MB) on the first run.

## Run

```bash
cd ~/FYP_SemCon/LivePortrait-encodec
python offline/run_offline.py --video ~/FYP_SemCon/files/testing_example.mp4 --max-seconds 20 --side-by-side
```

Options: `--audio X.wav` (separate speech file), `--avatar portrait.jpg`
(default: first video frame, i.e. self-reconstruction), `--fps 20`,
`--bandwidth 6` (EnCodec 1.5/3/6/12/24), `--audio-chunk-ms 240`,
`--audio-device cuda`, `--motion-loss 0.05`, `--no-crop`,
`--keep-original-audio`.

## Outputs (`offline/results/<video>_<time>/`)

| file | content |
|---|---|
| `output.mp4` | received avatar video + EnCodec-decoded audio |
| `side_by_side.mp4` | sender's face crop \| received avatar (`--side-by-side`) |
| `video_only.mp4` | the same video without audio |
| `audio_original_24k.wav`, `audio_encodec_24k.wav` | audio before / after EnCodec |
| `avatar_sent.jpg` | the portrait exactly as transmitted |
| `semantics.npz` | every motion packet (raw bytes), its frame index, decoded values |
| `summary.json` | bit rates (motion / audio / portrait / whole clip), packet counts, timings |
