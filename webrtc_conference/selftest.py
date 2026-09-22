# coding: utf-8
"""
End-to-end verification without a second machine, camera or microphone.

    python -m webrtc_conference.selftest

Stage 1  wire format round-trips
Stage 2  EnCodec encode -> pack -> unpack -> decode on synthetic speech
Stage 3  LivePortrait motion extract -> pack -> unpack -> render
Stage 4  two real aiortc peers over loopback, exchanging an avatar, a burst of
         motion vectors and a burst of audio through the actual datachannels
"""

import argparse
import asyncio
import os
import sys
import time

import cv2
import numpy as np

from . import protocol as P
from .audio_codec import SAMPLE_RATE, EncodecStreamCodec
from .motion_codec import (AvatarRenderer, LivePortraitEngine, MotionEncoder,
                           decode_avatar_jpeg, encode_avatar_jpeg, load_avatar_rgb)
from .peer import NeuralPeer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(REPO, "assets", "examples", "source", "s6.jpg")
DEFAULT_DRIVING = os.path.join(REPO, "assets", "examples", "source", "s9.jpg")

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = ""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


# ── stage 1 ───────────────────────────────────────────────────────────────────

def stage_protocol():
    print("\nStage 1: wire format")
    rng = np.random.default_rng(0)
    exp = (rng.standard_normal((21, 3)) * 0.05).astype(np.float32)
    t = np.array([0.12, -0.07, 0.0], dtype=np.float32)
    blob = P.pack_motion(65535, 999, 11.5, -22.25, 3.125, 1.037, t, exp, calibrate=True)
    m = P.unpack_motion(blob)
    check("motion packet is 150 bytes", len(blob) == P.MOTION_PACKET_SIZE == 150,
          f"{len(blob)} B")
    check("motion header survives", m.seq == 65535 and m.t_ms == 999 and m.calibrate)
    check("expression error under 1e-4", float(np.abs(m.exp - exp).max()) < 1e-4,
          f"max {np.abs(m.exp - exp).max():.2e}")

    codes = rng.integers(0, 1024, size=(8, 18))
    packed = P.pack_audio(5, codes)
    ap = P.unpack_audio(packed)
    check("10-bit code packing is lossless", np.array_equal(ap.codes, codes))
    check("audio payload size matches formula",
          len(packed) == P.audio_payload_size(8, 18), f"{len(packed)} B")

    payload = bytes(rng.integers(0, 256, size=30000, dtype=np.uint8))
    r, out = P.BlobReassembler(), None
    for chunk in P.pack_blob(3, payload):
        out = r.feed(chunk)
    check("blob reassembly", out == payload, f"{len(payload)} B in 4 chunks")


# ── stage 2 ───────────────────────────────────────────────────────────────────

def stage_audio(bandwidth: float, chunk_ms: int, device: str):
    print(f"\nStage 2: EnCodec  (bandwidth={bandwidth} kbps, chunk={chunk_ms} ms)")
    codec = EncodecStreamCodec(bandwidth=bandwidth, device=device, chunk_ms=chunk_ms)
    check("quantiser configured", codec.n_q > 0,
          f"n_q={codec.n_q}, nominal {codec.nominal_kbps:.1f} kbps")

    mic_sr = 48000
    n = int(mic_sr * chunk_ms / 1000)
    t = np.arange(n) / mic_sr
    speechy = (0.3 * np.sin(2 * np.pi * 140 * t)
               * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

    sizes, enc_ms, dec_ms = [], [], []
    for i in range(5):
        codes = codec.encode(speechy, mic_sr)
        wire = P.pack_audio(i, codes)
        pcm = codec.decode(P.unpack_audio(wire).codes)
        sizes.append(len(wire))
        enc_ms.append(codec.stats.encode_ms)
        dec_ms.append(codec.stats.decode_ms)

    wire_kbps = np.mean(sizes) * 8 / (chunk_ms / 1000) / 1000
    budget = chunk_ms
    check("wire bitrate near nominal", abs(wire_kbps - codec.nominal_kbps) < 1.5,
          f"{wire_kbps:.2f} kbps on the wire vs {codec.nominal_kbps:.2f} nominal")
    check("codec keeps up with real time",
          np.mean(enc_ms) + np.mean(dec_ms) < budget,
          f"encode {np.mean(enc_ms):.0f} ms + decode {np.mean(dec_ms):.0f} ms "
          f"per {chunk_ms} ms chunk")
    check("decoded audio is finite and non-silent",
          np.isfinite(pcm).all() and float(np.abs(pcm).max()) > 0.01,
          f"peak {np.abs(pcm).max():.3f}")


# ── stage 3 ───────────────────────────────────────────────────────────────────

def stage_video(source: str, driving: str, out_dir: str, cpu: bool):
    print("\nStage 3: LivePortrait motion codec")
    engine = LivePortraitEngine(force_cpu=cpu)
    print(f"  device: {engine.device}")

    src_rgb = load_avatar_rgb(source)
    jpeg = encode_avatar_jpeg(src_rgb)
    check("avatar JPEG round-trips",
          decode_avatar_jpeg(jpeg).shape[2] == 3, f"{len(jpeg) / 1024:.0f} KiB")

    t0 = time.perf_counter()
    renderer = AvatarRenderer(engine, decode_avatar_jpeg(jpeg))
    setup_ms = (time.perf_counter() - t0) * 1000
    check("renderer built from the received avatar", renderer.f_s is not None,
          f"one-time setup {setup_ms:.0f} ms")

    encoder = MotionEncoder(engine)
    drive_bgr = cv2.imread(driving)
    check("driving frame loaded", drive_bgr is not None, driving)
    if drive_bgr is None:
        return

    # First pass establishes the neutral reference, second produces motion.
    frames, extract_ms, render_ms = [], [], []
    for i in range(3):
        t0 = time.perf_counter()
        state = encoder.extract(drive_bgr)
        extract_ms.append((time.perf_counter() - t0) * 1000)
        if state is None:
            check("face found in driving frame", False)
            return
        wire = P.pack_motion(i, i * 50, state.pitch, state.yaw, state.roll,
                             state.scale, state.t, state.exp, calibrate=(i == 0))
        t0 = time.perf_counter()
        frames.append(renderer.render(P.unpack_motion(wire)))
        render_ms.append((time.perf_counter() - t0) * 1000)

    check("renderer produced 512x512 frames",
          frames[-1].shape == (512, 512, 3), str(frames[-1].shape))
    check("output is not blank", int(frames[-1].std()) > 5,
          f"std {frames[-1].std():.1f}")

    steady_extract = float(np.mean(extract_ms[1:]))
    steady_render = float(np.mean(render_ms[1:]))
    check("motion extraction sustains the default 20 fps send rate",
          steady_extract < 50, f"{steady_extract:.0f} ms/frame "
          f"(~{1000 / max(steady_extract, 1e-6):.0f} fps)")
    # Rendering is the real ceiling and it is per remote peer, since warp+decode
    # runs once for each participant on screen.
    check("rendering is usable in real time", steady_render < 200,
          f"{steady_render:.0f} ms/frame → ~{1000 / max(steady_render, 1e-6):.1f} fps "
          f"for one remote peer, ~{1000 / max(steady_render * 3, 1e-6):.1f} fps each "
          f"with three")

    os.makedirs(out_dir, exist_ok=True)
    panel = np.hstack([
        cv2.resize(renderer.source_bgr_256, (512, 512)),
        cv2.resize(encoder.last_crop_bgr, (512, 512)),
        frames[-1],
    ])
    path = os.path.join(out_dir, "selftest_video.png")
    cv2.imwrite(path, panel)
    print(f"  wrote {path}  [source | driving crop | reconstructed]")

    kbps = P.MOTION_PACKET_SIZE * 8 * 20 / 1000
    print(f"  video bitrate at 20 fps: {kbps:.1f} kbps "
          f"({P.MOTION_PACKET_SIZE} B/frame)")


# ── stage 4 ───────────────────────────────────────────────────────────────────

async def stage_transport(source: str):
    print("\nStage 4: aiortc loopback over real datachannels")
    a_to_b: asyncio.Queue = asyncio.Queue()
    b_to_a: asyncio.Queue = asyncio.Queue()
    received = {"avatar": None, "motion": [], "audio": [], "ctrl": []}

    async def send_a(_to, data):
        await a_to_b.put(data)

    async def send_b(_to, data):
        await b_to_a.put(data)

    alice = NeuralPeer("b", "Bob", True, send_a, ice_servers=[])
    bob = NeuralPeer("a", "Alice", False, send_b, ice_servers=[])

    bob.on_avatar = lambda pid, d: received.__setitem__("avatar", d)
    bob.on_motion = lambda pid, p: received["motion"].append(p)
    bob.on_audio = lambda pid, p: received["audio"].append(p)
    bob.on_ctrl = lambda pid, m: received["ctrl"].append(m)

    async def pump(q, peer):
        while True:
            await peer.handle_signal(await q.get())

    pumps = [asyncio.create_task(pump(a_to_b, bob)),
             asyncio.create_task(pump(b_to_a, alice))]

    await alice.start_offer()
    try:
        await asyncio.wait_for(
            asyncio.gather(alice.connected.wait(), bob.connected.wait()), timeout=20)
        check("peers connected", True,
              f"alice={alice.pc.connectionState} bob={bob.pc.connectionState}")
    except asyncio.TimeoutError:
        check("peers connected", False, "timed out after 20 s")
        for t in pumps:
            t.cancel()
        await alice.close()
        await bob.close()
        return

    await asyncio.sleep(0.3)
    check("all four channels open",
          all(c.readyState == "open"
              for c in (alice.ctrl, alice.blob, alice.motion, alice.audio)),
          ", ".join(f"{c.label}={c.readyState}"
                    for c in (alice.ctrl, alice.blob, alice.motion, alice.audio)))

    jpeg = encode_avatar_jpeg(load_avatar_rgb(source))
    alice.send_ctrl({"type": "hello", "name": "Alice"})
    alice.send_avatar(1, jpeg)

    rng = np.random.default_rng(1)
    n_motion = 50
    for i in range(n_motion):
        alice.send_motion(P.pack_motion(
            i, i * 50, 5.0, -3.0, 1.0, 1.0,
            np.zeros(3, np.float32),
            (rng.standard_normal((21, 3)) * 0.02).astype(np.float32),
            calibrate=(i == 0)))
        if i % 5 == 0:
            alice.send_audio(P.pack_audio(
                i // 5, rng.integers(0, 1024, size=(8, 18))))
        await asyncio.sleep(0.01)

    for _ in range(50):
        await asyncio.sleep(0.1)
        if (received["avatar"] and len(received["motion"]) >= n_motion
                and len(received["audio"]) >= 10):
            break

    check("avatar arrived intact over the reliable channel",
          received["avatar"] == jpeg, f"{len(jpeg) / 1024:.0f} KiB")
    check("ctrl hello arrived",
          any(m.get("type") == "hello" for m in received["ctrl"]))
    check("motion vectors arrived", len(received["motion"]) >= n_motion * 0.9,
          f"{len(received['motion'])}/{n_motion}")
    check("audio chunks arrived", len(received["audio"]) >= 9,
          f"{len(received['audio'])}/10")
    if received["motion"]:
        check("first motion packet carries the calibrate flag",
              received["motion"][0].calibrate)

    v_kbps, a_kbps = bob.stats_rx.kbps()
    print(f"  measured over the link: video {v_kbps:.1f} kbps, "
          f"audio {a_kbps:.1f} kbps (at the synthetic send rate)")

    for t in pumps:
        t.cancel()
    await alice.close()
    await bob.close()


# ── stage 5 ───────────────────────────────────────────────────────────────────

async def stage_signaling(port: int):
    """Two clients meeting through the real signaling server, as client.py does."""
    print("\nStage 5: room signaling and mesh setup")
    from aiohttp import web

    from .signaling import SignalingClient, SignalingServer

    server = SignalingServer()
    runner = web.AppRunner(server.build_app(), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    url = f"ws://127.0.0.1:{port}/ws"

    peers: dict[str, dict[str, NeuralPeer]] = {"alice": {}, "bob": {}}
    got_motion = asyncio.Event()
    got_avatar: dict[str, bytes] = {}
    avatar = bytes([0xFF, 0xD8]) + bytes(60000) + bytes([0xFF, 0xD9])   # stands in for the JPEG

    def make_client(tag: str, is_second: bool):
        sig = SignalingClient(url, "selftest", tag)

        async def add(pid, name, offerer):
            if pid in peers[tag]:
                return peers[tag][pid]
            peer = NeuralPeer(pid, name, offerer, sig.signal, ice_servers=[])
            peer.on_motion = lambda p, pkt: got_motion.set()
            peer.on_avatar = lambda p, data: got_avatar.__setitem__(tag, data)
            # Greet exactly as client.py does: on the channels opening, not on
            # the connection state, which runs a round trip too early.
            peer.on_open = lambda p: (peer.send_ctrl({"type": "hello", "name": tag}),
                                      peer.send_avatar(1, avatar))
            peers[tag][pid] = peer
            if offerer:
                await peer.start_offer()
            return peer

        sig.on_peers = lambda lst: asyncio.gather(
            *[add(p["id"], p["name"], True) for p in lst])
        sig.on_peer_joined = lambda p: add(p["id"], p["name"], False)
        sig.on_signal = lambda src, data: _route(tag, src, data, add)
        return sig

    async def _route(tag, src, data, add):
        peer = peers[tag].get(src) or await add(src, src, False)
        await peer.handle_signal(data)

    alice_sig = make_client("alice", False)
    await alice_sig.connect()
    alice_task = asyncio.create_task(alice_sig.run())
    await asyncio.sleep(0.3)

    bob_sig = make_client("bob", True)
    await bob_sig.connect()
    bob_task = asyncio.create_task(bob_sig.run())

    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if peers["alice"] and peers["bob"]:
                a = next(iter(peers["alice"].values()))
                b = next(iter(peers["bob"].values()))
                if a.connected.is_set() and b.connected.is_set():
                    break
            await asyncio.sleep(0.1)

        check("both clients discovered each other through the room",
              len(peers["alice"]) == 1 and len(peers["bob"]) == 1,
              f"alice sees {len(peers['alice'])}, bob sees {len(peers['bob'])}")

        connected = all(p.connected.is_set()
                        for side in peers.values() for p in side.values())
        check("mesh connection established via signaling", connected)

        if connected:
            for _ in range(50):
                await asyncio.sleep(0.1)
                if len(got_avatar) == 2:
                    break
            # The avatar is sent once per peer, so this is the check that a
            # participant shows up as a face rather than an empty tile.
            check("both avatars arrived after the greeting",
                  got_avatar.get("alice") == avatar and got_avatar.get("bob") == avatar,
                  f"received {sorted(got_avatar)}")

            bob_peer = next(iter(peers["bob"].values()))
            bob_peer.send_motion(P.pack_motion(
                0, 0, 1.0, 2.0, 3.0, 1.0,
                np.zeros(3, np.float32), np.zeros((21, 3), np.float32),
                calibrate=True))
            try:
                await asyncio.wait_for(got_motion.wait(), timeout=5)
                check("motion flows over the signalled connection", True)
            except asyncio.TimeoutError:
                check("motion flows over the signalled connection", False)
    finally:
        for side in peers.values():
            for p in side.values():
                await p.close()
        alice_task.cancel()
        bob_task.cancel()
        await alice_sig.close()
        await bob_sig.close()
        await runner.cleanup()


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="Self-test for the neural conference stack")
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--driving", default=DEFAULT_DRIVING)
    p.add_argument("--bandwidth", type=float, default=6.0)
    p.add_argument("--audio-chunk-ms", type=int, default=240)
    p.add_argument("--audio-device", default="cpu")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="selftest_out")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["protocol", "audio", "video", "transport", "signaling"])
    p.add_argument("--signaling-port", type=int, default=8799)
    args = p.parse_args(argv)

    if "protocol" not in args.skip:
        stage_protocol()
    if "audio" not in args.skip:
        stage_audio(args.bandwidth, args.audio_chunk_ms, args.audio_device)
    if "video" not in args.skip:
        stage_video(args.source, args.driving, args.out, args.cpu)
    if "transport" not in args.skip:
        asyncio.run(stage_transport(args.source))
    if "signaling" not in args.skip:
        asyncio.run(stage_signaling(args.signaling_port))

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: " + ", ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
