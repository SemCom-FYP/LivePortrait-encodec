# coding: utf-8
"""
Conference participant.

    python -m webrtc_conference.client --room demo --name Alice \
        --avatar assets/examples/source/s6.jpg

Threading: aiortc owns an asyncio loop on a background thread; OpenCV owns the
main thread. Torch work never touches the loop — capture, motion extraction,
rendering and audio coding each run on their own thread and hand finished
buffers across with `call_soon_threadsafe` or a one-deep mailbox.
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import protocol as P
from . import ui
from .audio_codec import SAMPLE_RATE, CrossfadeJoiner, EncodecStreamCodec
from .media import (CameraReader, LatestSlot, MicReader, SpeakerMixer,
                    list_devices, probe_input_rate)
from .motion_codec import (AvatarRenderer, LivePortraitEngine, MotionEncoder,
                           decode_avatar_jpeg, encode_avatar_jpeg, load_avatar_rgb)
from .peer import NeuralPeer, RateMeter
from .signaling import DEFAULT_PORT, SignalingClient

log = logging.getLogger("client")


@dataclass
class RemoteState:
    """Everything we keep about one other participant."""
    peer_id: str
    name: str
    renderer: Optional[AvatarRenderer] = None
    avatar_jpeg: Optional[bytes] = None
    status: str = "connecting…"
    pending_motion: LatestSlot = field(default_factory=LatestSlot)
    audio_q: "queue.Queue[P.AudioPacket]" = field(default_factory=lambda: queue.Queue(maxsize=8))
    joiner: CrossfadeJoiner = field(default_factory=CrossfadeJoiner)
    frames_rendered: int = 0
    render_ms: float = 0.0
    last_seq: int = -1
    lost: int = 0


class ConferenceApp:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.peers: dict[str, NeuralPeer] = {}
        self.remotes: dict[str, RemoteState] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.signaling: Optional[SignalingClient] = None

        self._motion_seq = 0
        self._audio_seq = 0
        self._calibrate_next = True
        self._tx_video = RateMeter()
        self._tx_audio = RateMeter()
        self._extract_ms = 0.0
        self._lock = threading.Lock()

        print("Loading LivePortrait networks …")
        self.engine = LivePortraitEngine(force_cpu=args.cpu)
        print(f"  device: {self.engine.device}")

        self.avatar_rgb = load_avatar_rgb(args.avatar)
        self.avatar_jpeg = encode_avatar_jpeg(self.avatar_rgb, quality=args.avatar_quality)
        print(f"  avatar: {args.avatar}  ({len(self.avatar_jpeg) / 1024:.0f} KiB on the wire, sent once)")

        self.encoder = MotionEncoder(self.engine, det_interval=args.det_interval,
                                     crop_driving=not args.no_crop)
        self.camera = CameraReader(args.camera, args.width, args.height)

        self.self_renderer: Optional[AvatarRenderer] = None
        if args.preview_self:
            print("  building self-preview renderer …")
            self.self_renderer = AvatarRenderer(self.engine, self.avatar_rgb,
                                                args.driving_multiplier)
        self.self_slot = LatestSlot()

        # Audio
        self.audio_codec: Optional[EncodecStreamCodec] = None
        self.speaker: Optional[SpeakerMixer] = None
        self.mic: Optional[MicReader] = None
        self._mic_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
        if not args.no_audio:
            print(f"  loading EnCodec ({args.bandwidth} kbps, {args.audio_device}) …")
            self.audio_codec = EncodecStreamCodec(
                bandwidth=args.bandwidth, device=args.audio_device,
                chunk_ms=args.audio_chunk_ms)
            self.decode_codec = EncodecStreamCodec(
                bandwidth=args.bandwidth, device=args.audio_device,
                chunk_ms=args.audio_chunk_ms)
            self.speaker = SpeakerMixer(SAMPLE_RATE, device=args.output_device)

    # ── threads ──────────────────────────────────────────────────────────────

    def start_workers(self):
        for target in (self._capture_loop, self._render_loop):
            threading.Thread(target=target, daemon=True).start()
        if self.audio_codec is not None:
            threading.Thread(target=self._audio_encode_loop, daemon=True).start()
            threading.Thread(target=self._audio_decode_loop, daemon=True).start()
            self.speaker.start()
            chunk_ms = self.args.audio_chunk_ms
            rate = probe_input_rate(self.args.input_device)
            self.mic = MicReader(
                self._on_mic_chunk,
                chunk_samples=int(rate * chunk_ms / 1000),
                device=self.args.input_device, sample_rate=rate)
            self.mic.start()
            print(f"  mic: {self.mic.name} @ {self.mic.sample_rate} Hz, "
                  f"{chunk_ms} ms chunks")

    def _capture_loop(self):
        """Webcam -> motion vector -> every peer."""
        period = 1.0 / max(self.args.fps, 1)
        next_tick = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(period, next_tick - now))
                continue
            next_tick = max(now, next_tick + period)

            frame = self.camera.read()
            if frame is None:
                continue

            t0 = time.perf_counter()
            state = self.encoder.extract(frame)
            self._extract_ms = (time.perf_counter() - t0) * 1000
            if state is None:
                self.self_slot.put(("noface", frame))
                continue

            calibrate = self._calibrate_next
            self._calibrate_next = False
            payload = P.pack_motion(
                self._motion_seq, int(time.monotonic() * 1000) & 0xFFFFFFFF,
                state.pitch, state.yaw, state.roll, state.scale, state.t, state.exp,
                calibrate=calibrate)
            self._motion_seq += 1
            # Counted once, not once per peer: this is the per-stream rate, the
            # number that is comparable to a conventional codec's bitrate.
            self._tx_video.add(len(payload))

            self._broadcast_motion(payload)
            self.self_slot.put(("ok", self.encoder.last_crop_bgr))

            if self.self_renderer is not None:
                try:
                    pkt = P.unpack_motion(payload)
                    self.self_renderer.render(pkt)
                except Exception as exc:
                    log.debug("self preview failed: %s", exc)

    def _broadcast_motion(self, payload: bytes):
        if self.loop is None:
            return
        for peer in list(self.peers.values()):
            self.loop.call_soon_threadsafe(_swallow, peer.send_motion, payload)

    def _on_mic_chunk(self, pcm: np.ndarray):
        try:
            self._mic_q.put_nowait(pcm)
        except queue.Full:
            pass                                    # mic outran the encoder; drop

    def _audio_encode_loop(self):
        while self.running:
            try:
                pcm = self._mic_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                codes = self.audio_codec.encode(pcm, self.mic.sample_rate)
            except Exception as exc:
                log.warning("audio encode failed: %s", exc)
                continue
            payload = P.pack_audio(self._audio_seq, codes)
            self._audio_seq += 1
            self._tx_audio.add(len(payload))
            if self.loop is not None:
                for peer in list(self.peers.values()):
                    self.loop.call_soon_threadsafe(_swallow, peer.send_audio, payload)

    def _audio_decode_loop(self):
        while self.running:
            work = False
            for remote in list(self.remotes.values()):
                try:
                    pkt = remote.audio_q.get_nowait()
                except queue.Empty:
                    continue
                work = True
                try:
                    pcm = self.decode_codec.decode(pkt.codes)
                except Exception as exc:
                    log.debug("audio decode failed: %s", exc)
                    continue
                self.speaker.push(remote.peer_id, remote.joiner.push(pcm))
            if not work:
                time.sleep(0.005)

    def _render_loop(self):
        """Round-robins the GPU across every peer that has a fresh motion vector.

        Warp+decode is the most expensive step in the whole system and it runs
        once per participant on screen, so peers take strict turns. Each one
        renders only its newest vector; anything that arrived while the GPU was
        busy elsewhere is dropped rather than queued, which keeps latency flat
        as the room grows instead of letting a backlog build.
        """
        min_period = 1.0 / self.args.max_render_fps if self.args.max_render_fps else 0.0
        last_render: dict[str, float] = {}
        while self.running:
            work = False
            for remote in list(self.remotes.values()):
                if remote.renderer is None:
                    continue
                now = time.perf_counter()
                if now - last_render.get(remote.peer_id, 0.0) < min_period:
                    continue
                pkt = remote.pending_motion.take()
                if pkt is None:
                    continue
                work = True
                last_render[remote.peer_id] = now
                try:
                    remote.renderer.render(pkt)
                except Exception as exc:
                    log.warning("[%s] render failed: %s", remote.name, exc)
                    continue
                remote.render_ms = (time.perf_counter() - now) * 1000
                remote.frames_rendered += 1
            if not work:
                time.sleep(0.003)

    def _build_renderer(self, remote: RemoteState, jpeg: bytes):
        """Off-loop: decode the peer's portrait and pay the one-time F/W setup."""
        try:
            rgb = decode_avatar_jpeg(jpeg)
            renderer = AvatarRenderer(self.engine, rgb, self.args.driving_multiplier)
        except Exception as exc:
            remote.status = f"avatar failed: {exc}"
            log.warning("[%s] %s", remote.name, remote.status)
            return
        remote.renderer = renderer
        remote.status = ""
        log.info("[%s] avatar ready", remote.name)

    # ── peer lifecycle ───────────────────────────────────────────────────────

    def _attach_handlers(self, peer: NeuralPeer):
        remote = self.remotes.setdefault(
            peer.peer_id, RemoteState(peer.peer_id, peer.name))

        def on_motion(pid, pkt):
            r = self.remotes.get(pid)
            if r is None:
                return
            if r.last_seq >= 0:
                gap = (pkt.seq - r.last_seq) & 0xFFFF
                if 1 < gap < 1000:
                    r.lost += gap - 1
            r.last_seq = pkt.seq
            r.pending_motion.put(pkt)

        def on_audio(pid, pkt):
            r = self.remotes.get(pid)
            if r is None:
                return
            try:
                r.audio_q.put_nowait(pkt)
            except queue.Full:
                pass

        def on_avatar(pid, data):
            r = self.remotes.get(pid)
            if r is None:
                return
            r.avatar_jpeg = data
            r.status = "building avatar…"
            threading.Thread(target=self._build_renderer, args=(r, data),
                             daemon=True).start()

        def on_ctrl(pid, msg):
            r = self.remotes.get(pid)
            if r is None:
                return
            if msg.get("type") == "hello":
                r.name = msg.get("name", r.name)
                log.info("[%s] hello (%s)", r.name, msg.get("client", "?"))
            elif msg.get("type") == "bye":
                r.status = "left"

        def on_state(pid, state):
            r = self.remotes.get(pid)
            if r is None:
                return
            if state == "connected":
                if r.renderer is None:
                    r.status = "waiting for avatar…"
                self._greet(pid)
            elif state in ("failed", "closed", "disconnected"):
                r.status = state

        peer.on_motion = on_motion
        peer.on_audio = on_audio
        peer.on_avatar = on_avatar
        peer.on_ctrl = on_ctrl
        peer.on_state = on_state
        return remote

    def _greet(self, peer_id: str):
        peer = self.peers.get(peer_id)
        if peer is None:
            return
        peer.send_ctrl({"type": "hello", "name": self.args.name,
                        "client": "liveportrait-neural-conf",
                        "bandwidth": self.args.bandwidth})
        peer.send_avatar(1, self.avatar_jpeg)

    async def _add_peer(self, peer_id: str, name: str, is_offerer: bool):
        if peer_id in self.peers:
            return self.peers[peer_id]
        peer = NeuralPeer(peer_id, name, is_offerer, self.signaling.signal,
                          ice_servers=self.args.stun)
        self.peers[peer_id] = peer
        self._attach_handlers(peer)
        if is_offerer:
            await peer.start_offer()
        return peer

    async def _remove_peer(self, peer_id: str):
        peer = self.peers.pop(peer_id, None)
        self.remotes.pop(peer_id, None)
        if self.speaker is not None:
            self.speaker.drop(peer_id)
        if peer is not None:
            await peer.close()

    # ── asyncio side ─────────────────────────────────────────────────────────

    async def _async_main(self):
        self.loop = asyncio.get_running_loop()
        url = self.args.signaling
        self.signaling = SignalingClient(url, self.args.room, self.args.name)

        async def on_peers(peers):
            print(f"Joined room '{self.args.room}' as {self.args.name} "
                  f"({len(peers)} already present)")
            for p in peers:
                await self._add_peer(p["id"], p["name"], is_offerer=True)

        async def on_peer_joined(p):
            print(f"{p['name']} joined")
            await self._add_peer(p["id"], p["name"], is_offerer=False)

        async def on_peer_left(pid):
            name = self.remotes[pid].name if pid in self.remotes else pid
            print(f"{name} left")
            await self._remove_peer(pid)

        async def on_signal(src, data):
            peer = self.peers.get(src)
            if peer is None:
                peer = await self._add_peer(src, src, is_offerer=False)
            await peer.handle_signal(data)

        self.signaling.on_peers = on_peers
        self.signaling.on_peer_joined = on_peer_joined
        self.signaling.on_peer_left = on_peer_left
        self.signaling.on_signal = on_signal

        await self.signaling.connect()
        runner = asyncio.create_task(self.signaling.run())
        while self.running:
            await asyncio.sleep(0.1)
        runner.cancel()

        for pid in list(self.peers):
            await self._remove_peer(pid)
        await self.signaling.close()

    # ── UI (main thread) ─────────────────────────────────────────────────────

    def ui_loop(self):
        window = "Neural Conference — LivePortrait + EnCodec"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        ref_kbps = ui.h264_reference_kbps(self.args.width, self.args.height, self.args.fps)

        while self.running:
            tiles = [self._self_tile()]
            for remote in list(self.remotes.values()):
                tiles.append(self._remote_tile(remote))

            tx_v, tx_a = self._tx_video.kbps(), self._tx_audio.kbps()
            header =(f"room '{self.args.room}'   peers {len(self.remotes)}   "
                      f"TX/stream: video {tx_v:5.1f} kbps + audio {tx_a:4.1f} kbps   "
                      f"(H.264 at {self.args.width}x{self.args.height} would be "
                      f"~{ref_kbps:.0f} kbps)")
            footer = "q quit    r re-calibrate neutral pose    s save a frame"

            cv2.imshow(window, ui.compose(tiles, self.args.tile, header, footer))
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                self.running = False
            elif key == ord("r"):
                self._calibrate_next = True
                self.encoder.reset_tracking()
                for r in self.remotes.values():
                    if r.renderer is not None:
                        r.renderer.reset_reference()
                print("Re-calibrating neutral pose.")
            elif key == ord("s"):
                path = f"conference_{int(time.time())}.png"
                cv2.imwrite(path, ui.compose(tiles, self.args.tile, header, footer))
                print(f"Saved {path}")

        cv2.destroyAllWindows()

    def _self_tile(self) -> ui.Tile:
        slot = self.self_slot.peek()
        status, image = slot if slot else ("waiting", None)
        if self.self_renderer is not None and status == "ok":
            image = self.self_renderer.last_frame_bgr
        # Derived from the same meter as the bitrate, so the two always agree —
        # frames dropped for a missing face show up in both.
        subtitle = (f"{self._tx_video.pps():4.1f} fps sent   "
                    f"extract {self._extract_ms:.0f} ms   "
                    f"{P.MOTION_PACKET_SIZE} B/frame")
        return ui.Tile(
            title=f"{self.args.name} (you)",
            image=image,
            subtitle=subtitle,
            status="no face detected" if status == "noface" else "starting camera…",
            accent=ui.GREEN if status == "ok" else ui.RED,
        )

    def _remote_tile(self, remote: RemoteState) -> ui.Tile:
        peer = self.peers.get(remote.peer_id)
        image = remote.renderer.last_frame_bgr if remote.renderer is not None else None
        if remote.renderer is not None and remote.frames_rendered == 0:
            image = remote.renderer.source_bgr_256

        if peer is not None:
            v_kbps, a_kbps = peer.stats_rx.kbps()
            subtitle = (f"video {v_kbps:5.1f} kbps   audio {a_kbps:4.1f} kbps   "
                        f"render {remote.render_ms:.0f} ms   lost {remote.lost}")
        else:
            subtitle = ""
        return ui.Tile(title=remote.name, image=image, subtitle=subtitle,
                       status=remote.status or "waiting…",
                       accent=ui.GREEN if remote.renderer is not None else ui.AMBER)

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        thread = threading.Thread(
            target=lambda: asyncio.run(self._async_main()), daemon=True)
        thread.start()
        self.start_workers()
        try:
            self.ui_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            thread.join(timeout=5.0)
            self.camera.close()
            if self.mic is not None:
                self.mic.close()
            if self.speaker is not None:
                self.speaker.close()
            print("Left the room.")


def _swallow(fn, *a):
    try:
        fn(*a)
    except Exception as exc:
        log.debug("send failed: %s", exc)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Neural web conference: LivePortrait motion vectors + EnCodec over aiortc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--avatar", "-a", help="Portrait image that represents you")
    p.add_argument("--name", "-n", default=os.environ.get("USERNAME", "guest"))
    p.add_argument("--room", "-r", default="demo")
    p.add_argument("--signaling", default=f"ws://127.0.0.1:{DEFAULT_PORT}/ws")
    p.add_argument("--stun", nargs="*", default=["stun:stun.l.google.com:19302"],
                   help="STUN/TURN URLs (pass with no values to disable)")

    g = p.add_argument_group("video")
    g.add_argument("--camera", "-c", type=int, default=0)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--fps", type=int, default=20, help="Motion vectors sent per second")
    g.add_argument("--det-interval", type=int, default=30,
                   help="Full face detection every N frames; landmark tracking between")
    g.add_argument("--no-crop", action="store_true",
                   help="Skip face detection (fastest; needs the face to fill the frame)")
    g.add_argument("--driving-multiplier", type=float, default=1.0)
    g.add_argument("--max-render-fps", type=float, default=0,
                   help="Cap the render rate per remote peer (0 = as fast as the GPU allows)")
    g.add_argument("--preview-self", action="store_true",
                   help="Also render your own avatar locally (costs a second GPU pass)")
    g.add_argument("--avatar-quality", type=int, default=90)
    g.add_argument("--tile", type=int, default=384, help="Tile size in the grid")
    g.add_argument("--cpu", action="store_true", help="Force LivePortrait onto the CPU")

    g = p.add_argument_group("audio")
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--bandwidth", type=float, default=6.0,
                   choices=[1.5, 3.0, 6.0, 12.0, 24.0],
                   help="EnCodec target bandwidth in kbps")
    g.add_argument("--audio-chunk-ms", type=int, default=240,
                   help="Chunk length; lower means less latency and more overhead")
    g.add_argument("--audio-device", default="cpu",
                   help="Torch device for EnCodec ('cpu' keeps the GPU for rendering)")
    g.add_argument("--input-device", type=int, default=None)
    g.add_argument("--output-device", type=int, default=None)
    g.add_argument("--list-devices", action="store_true")

    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-10s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)

    if args.list_devices:
        print(list_devices())
        return 0
    if not args.avatar:
        print("error: --avatar is required (a portrait image that represents you)",
              file=sys.stderr)
        return 2
    if not os.path.exists(args.avatar):
        print(f"error: avatar not found: {args.avatar}", file=sys.stderr)
        return 2

    ConferenceApp(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
