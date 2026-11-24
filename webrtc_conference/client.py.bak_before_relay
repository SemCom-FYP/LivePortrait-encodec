# coding: utf-8
"""
Conference participant.

    python -m webrtc_conference.client --room demo --name Alice \
        --avatar assets/examples/source/s6.jpg

Threading: aiortc owns an asyncio loop on a background thread; Qt (PySide6)
owns the main thread. Torch work never touches the loop — capture, motion extraction,
rendering and audio coding each run on their own thread and hand finished
buffers across with `call_soon_threadsafe` or a one-deep mailbox.

A/V sync: audio is the master clock, because the speaker drains at a rate we
do not control. Motion and audio both carry the sender's capture timestamp,
and motion waits in a per-peer queue until the instant it was captured comes
up on that peer's audio playhead — see AVSync and MotionJitterBuffer.

Translation (T in the call, or --translate es): each peer's decoded voice
goes to a PeerTranslation instead of the speaker, and its translated speech
plays in place of the original — see translator.py. The motion clock keeps
running as if the original were playing, so faces stay live and the
translated voice follows a phrase behind.
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

import numpy as np
from aiortc import RTCIceServer

from . import protocol as P
from . import ui
from .audio_codec import SAMPLE_RATE, CrossfadeJoiner, EncodecStreamCodec
from .media import (AVSync, CameraReader, LatestSlot, MicReader,
                    MotionJitterBuffer, SpeakerMixer, list_devices,
                    probe_input_rate)
from .motion_codec import (AvatarRenderer, LivePortraitEngine, MotionEncoder,
                           decode_avatar_jpeg, encode_avatar_jpeg, load_avatar_rgb)
from .peer import DEFAULT_STUN, NeuralPeer, RateMeter, build_ice_servers
from .signaling import DEFAULT_PORT, SignalingClient
from .translator import (DEFAULT_POCKET_VOICE, LANGUAGES, VOICE_BUFFER_S, VOICE_SUFFIX,
                         PeerTranslation, PocketTTSVoice, SpeechTranslator)

log = logging.getLogger("client")

WINDOW_TITLE = "Neural Conference"
DEFAULT_SIGNALING = f"ws://127.0.0.1:{DEFAULT_PORT}/ws"
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "assets", "examples", "source")

# How far ahead of the audio playhead a motion packet may be and still be
# rendered now instead of waiting another tick. Kept small because it lands
# entirely on the leading side, and video that runs ahead of a voice is far
# more noticeable than video that trails it (ITU-R BT.1359 puts the bounds at
# roughly +45 ms against -125 ms).
AV_SYNC_SLACK_MS = 10

# Cycle for the "Render" control (D): 1 decodes every due packet, N>1 decodes
# only every Nth and reuses the previous frame otherwise — see webcam_inference.py's
# --decode_every, which this mirrors. Trades a peer's own update rate for GPU time
# the round-robin can spend on the rest of the room (and on self-preview).
DECODE_EVERY_LEVELS = [1, 2, 3, 4]


@dataclass
class RemoteState:
    """Everything we keep about one other participant."""
    peer_id: str
    name: str
    renderer: Optional[AvatarRenderer] = None
    avatar_jpeg: Optional[bytes] = None
    status: str = "connecting…"
    pending_motion: MotionJitterBuffer = field(default_factory=MotionJitterBuffer)
    audio_q: "queue.Queue[P.AudioPacket]" = field(default_factory=lambda: queue.Queue(maxsize=8))
    joiner: CrossfadeJoiner = field(default_factory=CrossfadeJoiner)
    av_sync: AVSync = field(default_factory=AVSync)
    translation: Optional[PeerTranslation] = None
    sync_drift_ms: float = 0.0
    frames_rendered: int = 0
    render_ms: float = 0.0
    last_seq: int = -1
    lost: int = 0
    due_count: int = 0           # due packets seen, for the decode-every cadence


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
        self.decode_every = (args.decode_every
                             if args.decode_every in DECODE_EVERY_LEVELS else 1)
        self._self_due_count = 0

        self.ice_servers = _ice_servers(args)
        print(f"  ICE: {_describe_ice(self.ice_servers)}")

        print("Loading LivePortrait networks …")
        self.engine = LivePortraitEngine(force_cpu=args.cpu)
        print(f"  device: {self.engine.device}")

        self.avatar_rgb = load_avatar_rgb(args.avatar)
        self.avatar_jpeg = encode_avatar_jpeg(self.avatar_rgb, quality=args.avatar_quality)
        print(f"  avatar: {args.avatar}  ({len(self.avatar_jpeg) / 1024:.0f} KiB on the wire, sent once)")

        self.encoder = MotionEncoder(self.engine, det_interval=args.det_interval,
                                     crop_driving=not args.no_crop)
        # --camera -1 skips the webcam entirely: useful for a listen-only
        # client testing translation on a machine with only one camera, since
        # a peer still gets this client's audio and avatar either way — only
        # its own motion (and so its rendered tile) never appears.
        self.camera = (CameraReader(args.camera, args.width, args.height)
                       if args.camera >= 0 else None)

        # Your own avatar, rendered from your own motion vectors exactly as the
        # other side will render it — so you see what they see. Switchable in
        # the call; the renderer is built on first use if it starts off.
        self.self_renderer: Optional[AvatarRenderer] = None
        self.self_preview = False                   # rendering your own avatar right now
        self._self_building = False
        self._notices: "queue.Queue[str]" = queue.Queue()
        if args.preview_self:
            print("  building self-preview renderer …")
            self._build_self_renderer()
        self.self_slot = LatestSlot()
        self.self_motion = LatestSlot()             # newest own packet, for the self render
        self.self_view = "avatar"                   # which one fills the tile: avatar | camera
        self._self_frames = 0
        self._self_render_ms = 0.0

        # Audio
        self.audio_codec: Optional[EncodecStreamCodec] = None
        self.speaker: Optional[SpeakerMixer] = None
        self.mic: Optional[MicReader] = None
        self._mic_q: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=4)
        if not args.no_audio:
            print(f"  loading EnCodec ({args.bandwidth} kbps, {args.audio_device}) …")
            self.audio_codec = EncodecStreamCodec(
                bandwidth=args.bandwidth, device=args.audio_device,
                chunk_ms=args.audio_chunk_ms)
            self.decode_codec = EncodecStreamCodec(
                bandwidth=args.bandwidth, device=args.audio_device,
                chunk_ms=args.audio_chunk_ms)
            self.speaker = SpeakerMixer(SAMPLE_RATE, device=args.output_device)

        # Translation: built on first use, since Whisper and each language's
        # models take seconds to load (and a download the first time).
        self.translator: Optional[SpeechTranslator] = None
        self.translate_to: Optional[str] = None     # language being heard now
        self._translate_loading: Optional[str] = None
        # Only used when --tts pocket: one cloned voice, shared across every
        # peer's translated speech and every target language (see
        # translator.PocketTTSVoice — one checkpoint per language, all
        # cloned to the same reference clip).
        self.pocket_voice: Optional[PocketTTSVoice] = None
        if args.translate and self.audio_codec is not None:
            print(f"  loading translation to {LANGUAGES[args.translate].name} …")
            self._load_translation(args.translate)

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
        if self.camera is None:
            return
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

            # Stamped at the grab, not after extraction: the timestamp has to
            # mark when the pose happened, or the far side lines its render up
            # against our extraction time and the face trails the voice.
            cap_t_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF

            t0 = time.perf_counter()
            state = self.encoder.extract(frame)
            self._extract_ms = (time.perf_counter() - t0) * 1000
            if state is None:
                self.self_slot.put(("noface", frame))
                continue

            calibrate = self._calibrate_next
            self._calibrate_next = False
            payload = P.pack_motion(
                self._motion_seq, cap_t_ms,
                state.pitch, state.yaw, state.roll, state.scale, state.t, state.exp,
                calibrate=calibrate)
            self._motion_seq += 1
            # Counted once, not once per peer: this is the per-stream rate, the
            # number that is comparable to a conventional codec's bitrate.
            self._tx_video.add(len(payload))

            self._broadcast_motion(payload)
            self.self_slot.put(("ok", self.encoder.last_crop_bgr))
            if self.self_preview:
                # Rendered on the render thread, not here: a warp on this
                # thread would slow down the motion everyone else receives.
                self.self_motion.put(payload)

    def _broadcast_motion(self, payload: bytes):
        if self.loop is None:
            return
        for peer in list(self.peers.values()):
            self.loop.call_soon_threadsafe(_swallow, peer.send_motion, payload)

    def _on_mic_chunk(self, pcm: np.ndarray):
        # The callback only fires once the whole block has been captured, so
        # "now" is the *end* of the window. The wire timestamp marks the first
        # sample instead, which is the one the far side hears first — stamping
        # it "now" would hand every chunk to the receiver a whole chunk late
        # and let video run that far ahead of the voice.
        span_ms = pcm.size / self.mic.sample_rate * 1000
        t_ms = int(time.monotonic() * 1000 - span_ms) & 0xFFFFFFFF
        try:
            self._mic_q.put_nowait((t_ms, pcm))
        except queue.Full:
            pass                                    # mic outran the encoder; drop

    def _audio_encode_loop(self):
        while self.running:
            try:
                t_ms, pcm = self._mic_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                codes = self.audio_codec.encode(pcm, self.mic.sample_rate)
            except Exception as exc:
                log.warning("audio encode failed: %s", exc)
                continue
            payload = P.pack_audio(self._audio_seq, t_ms, codes)
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
                if self.translate_to is not None:
                    self._feed_translation(remote, pkt.t_ms, remote.joiner.push(pcm))
                    continue
                # Measured before the push, so it counts only what has to play
                # out first; the device's own buffering sits behind that again
                # and is large enough on Windows to matter for lip sync.
                delay_ms = (self.speaker.backlog_ms(remote.peer_id)
                            + self.speaker.output_latency_ms)
                remote.av_sync.note_audio_push(pkt.t_ms, delay_ms)
                self.speaker.push(remote.peer_id, remote.joiner.push(pcm))
            if not work:
                time.sleep(0.005)

    def _feed_translation(self, remote: RemoteState, t_ms: int, pcm: np.ndarray):
        if remote.translation is None:
            key = remote.peer_id + VOICE_SUFFIX
            remote.translation = PeerTranslation(
                self.translator, remote.name, self.translate_to,
                out=lambda p, k=key: self.speaker.push(k, p, cap_s=VOICE_BUFFER_S),
                backlog_ms=lambda k=key: self.speaker.backlog_ms(k),
                speak_fn=self.pocket_voice.speak if self.pocket_voice is not None else None)
        remote.translation.feed(pcm)
        # Nothing of the original is queued for this peer, so run the motion
        # clock as if it were playing with no backlog: the face stays live and
        # the translated voice follows a phrase behind it.
        remote.av_sync.note_audio_push(t_ms, self.speaker.output_latency_ms)

    def _load_translation(self, code: str) -> bool:
        """Whisper once, then this language's MT model and voice."""
        try:
            if self.translator is None:
                self.translator = SpeechTranslator(
                    self.args.asr_model, device=self.args.asr_device,
                    threads=self.args.translate_threads, mt_size=self.args.mt_model,
                    load_tts=(self.args.tts == "piper"))
                log.info("speech recognition: whisper %s on %s",
                         self.args.asr_model, self.translator.asr_device)
            if self.args.tts == "pocket" and self.pocket_voice is None:
                print(f"  loading Pocket TTS ({self.args.pocket_device}) …")
                self.pocket_voice = PocketTTSVoice(
                    device=self.args.pocket_device, quantize=self.args.pocket_quantize)
                self.pocket_voice.clone(self.args.voice_reference or DEFAULT_POCKET_VOICE)
            self.translator.prepare(code)
        except Exception as exc:
            self._notices.put(f"Can't translate to {LANGUAGES[code].name}: {exc}")
            log.warning("translation unavailable: %s", exc)
            return False
        finally:
            self._translate_loading = None
        self._set_translation(code)
        return True

    def _set_translation(self, code: Optional[str]):
        """Switch every peer to `code`, or back to their own voices with None."""
        self.translate_to = code
        for remote in list(self.remotes.values()):
            if remote.translation is not None:
                remote.translation.set_target(code)
            # Whichever voice is going out is cut now rather than finishing
            # the backlog it already had queued.
            self.speaker.drop(remote.peer_id + VOICE_SUFFIX if code is None
                              else remote.peer_id)

    def cycle_translation(self):
        """T in the call: off, then each --translate-langs language, then off."""
        if self.speaker is None:
            self._notices.put("Translation needs audio (started with --no-audio)")
            return
        if self._translate_loading:
            return
        order = [None] + list(self.args.translate_langs)
        current = self.translate_to if self.translate_to in order else None
        code = order[(order.index(current) + 1) % len(order)]
        if code is None:
            self._set_translation(None)
            self._notices.put("Translation off — original voices")
        elif self.translator is not None and self.translator.is_ready(code):
            self._set_translation(code)
            self._notices.put(f"Translating to {LANGUAGES[code].name}")
        else:
            self._translate_loading = code
            self._notices.put(f"Preparing {LANGUAGES[code].name} translation…")

            def build():
                if self._load_translation(code):
                    self._notices.put(f"Translating to {LANGUAGES[code].name}")
            threading.Thread(target=build, daemon=True).start()

    def cycle_decode_every(self):
        """D in the call: 1 -> 2 -> 3 -> 4 -> back to 1 (full rate)."""
        idx = DECODE_EVERY_LEVELS.index(self.decode_every)
        self.decode_every = DECODE_EVERY_LEVELS[(idx + 1) % len(DECODE_EVERY_LEVELS)]
        label = ("full rate" if self.decode_every == 1
                 else f"1/{self.decode_every} rate")
        self._notices.put(f"Render: {label} (more GPU time for the rest of the room)")

    def _render_loop(self):
        """Round-robins the GPU across every peer that has a fresh motion vector.

        Warp+decode is the most expensive step in the whole system and it runs
        once per participant on screen, so peers take strict turns. Each one
        renders a single vector per turn and everything the GPU missed while it
        was busy elsewhere is discarded, so a slow GPU costs frames rather than
        building a backlog and letting latency grow with the room.

        Which vector that is comes from the peer's audio playhead rather than
        from arrival order: the buffer hands back the newest packet whose
        capture instant has actually reached the speaker, so lip motion lands
        with the voice instead of a chunk ahead of it. The pick aims at where
        the playhead will be once this render finishes rather than where it is
        now — on CPU a warp costs more than a hundred milliseconds, which is
        the difference between lip sync and a face a beat behind the voice.
        """
        min_period = 1.0 / self.args.max_render_fps if self.args.max_render_fps else 0.0
        last_render: dict[str, float] = {}
        while self.running:
            work = self._render_self(last_render, min_period)
            for remote in list(self.remotes.values()):
                if remote.renderer is None:
                    continue
                now = time.perf_counter()
                if now - last_render.get(remote.peer_id, 0.0) < min_period:
                    continue
                playhead = remote.av_sync.playhead_t_ms()
                pkt = remote.pending_motion.take_due(
                    playhead, AV_SYNC_SLACK_MS + remote.render_ms)
                if pkt is None:
                    continue
                work = True
                last_render[remote.peer_id] = now
                remote.due_count += 1
                if remote.due_count % self.decode_every != 0:
                    continue        # tile keeps showing renderer.last_frame_bgr
                try:
                    remote.renderer.render(pkt)
                except Exception as exc:
                    log.warning("[%s] render failed: %s", remote.name, exc)
                    continue
                remote.render_ms = (time.perf_counter() - now) * 1000
                remote.frames_rendered += 1
                # Read again now the frame is actually on screen: this is the
                # error a viewer would see, not the one we aimed for.
                shown = remote.av_sync.playhead_t_ms()
                if shown is not None:
                    remote.sync_drift_ms = P.diff_t_ms(pkt.t_ms, int(shown))
            if not work:
                time.sleep(0.003)

    def _render_self(self, last_render: dict, min_period: float) -> bool:
        """One turn for your own avatar. Not paced to audio: there is none to
        wait for locally, so it simply shows the newest vector you sent."""
        if not self.self_preview or self.self_renderer is None:
            return False
        now = time.perf_counter()
        if now - last_render.get("", 0.0) < min_period:
            return False
        payload = self.self_motion.take()
        if payload is None:
            return False
        last_render[""] = now
        self._self_due_count += 1
        if self._self_due_count % self.decode_every != 0:
            return True          # tile keeps showing self_renderer.last_frame_bgr
        try:
            self.self_renderer.render(P.unpack_motion(payload))
        except Exception as exc:
            log.debug("self render failed: %s", exc)
            return True
        self._self_render_ms = (time.perf_counter() - now) * 1000
        self._self_frames += 1
        return True

    def _build_self_renderer(self) -> bool:
        """Pay the one-time F/W setup for your own portrait (~2 s on a GPU)."""
        try:
            self.self_renderer = AvatarRenderer(self.engine, self.avatar_rgb,
                                                self.args.driving_multiplier)
        except Exception as exc:
            self._notices.put(f"Can't preview your avatar: {exc}")
            log.warning("self preview unavailable: %s", exc)
            return False
        finally:
            self._self_building = False
        self.self_preview = True
        return True

    def set_self_preview(self, enabled: bool):
        """Turn rendering of your own avatar on or off mid-call."""
        if not enabled:
            self.self_preview = False
            self.self_motion.take()                 # drop anything still queued
            self._notices.put("Self preview off — your camera is shown instead")
        elif self.self_renderer is not None:
            self.self_preview = True
            self._notices.put("Self preview on")
        elif not self._self_building:
            self._self_building = True
            self._notices.put("Preparing your avatar preview…")

            def build():
                if self._build_self_renderer():
                    self._notices.put("Self preview on")
            threading.Thread(target=build, daemon=True).start()

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
                    r.status = "opening channels…"
            elif state in ("failed", "closed", "disconnected"):
                r.status = state

        def on_open(pid):
            r = self.remotes.get(pid)
            if r is not None and r.renderer is None:
                r.status = "waiting for avatar…"
            self._greet(pid)

        peer.on_motion = on_motion
        peer.on_audio = on_audio
        peer.on_avatar = on_avatar
        peer.on_ctrl = on_ctrl
        peer.on_state = on_state
        peer.on_open = on_open
        return remote

    def _greet(self, peer_id: str, attempt: int = 0):
        """Send our name and portrait, once the channels can actually carry them.

        The avatar goes out exactly once per peer, so a send that lands before
        the channel opens is not a dropped frame — it is a participant who stays
        a grey tile for the rest of the call. Hence the retry.
        """
        peer = self.peers.get(peer_id)
        if peer is None or peer.closed:
            return
        if peer.ctrl.readyState != "open" or peer.blob.readyState != "open":
            if attempt >= 40:                       # ~20 s
                r = self.remotes.get(peer_id)
                if r is not None:
                    r.status = "channels never opened"
                log.warning("[%s] giving up on the greeting: ctrl=%s blob=%s",
                            peer.name, peer.ctrl.readyState, peer.blob.readyState)
            elif self.loop is not None:
                self.loop.call_later(0.5, self._greet, peer_id, attempt + 1)
            return
        peer.send_ctrl({"type": "hello", "name": self.args.name,
                        "client": "liveportrait-neural-conf",
                        "bandwidth": self.args.bandwidth})
        peer.send_avatar(1, self.avatar_jpeg)
        log.info("[%s] sent hello + avatar (%.0f KiB)",
                 peer.name, len(self.avatar_jpeg) / 1024)

    async def _add_peer(self, peer_id: str, name: str, is_offerer: bool):
        if peer_id in self.peers:
            return self.peers[peer_id]
        peer = NeuralPeer(peer_id, name, is_offerer, self.signaling.signal,
                          ice_servers=self.ice_servers)
        self.peers[peer_id] = peer
        self._attach_handlers(peer)
        if is_offerer:
            await peer.start_offer()
        return peer

    async def _remove_peer(self, peer_id: str):
        peer = self.peers.pop(peer_id, None)
        remote = self.remotes.pop(peer_id, None)
        if remote is not None and remote.translation is not None:
            remote.translation.close()
        if self.speaker is not None:
            self.speaker.drop(peer_id)
            self.speaker.drop(peer_id + VOICE_SUFFIX)
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
        """Run the call window on the main thread until the user leaves."""
        ref_kbps = ui.h264_reference_kbps(self.args.width, self.args.height, self.args.fps)
        win: Optional[ui.CallWindow] = None

        def refresh() -> Optional[ui.CallState]:
            if not self.running:
                win.close()
                return None
            tiles = [self._self_tile()]
            for remote in list(self.remotes.values()):
                tiles.append(self._remote_tile(remote))
            tx_v, tx_a = self._tx_video.kbps(), self._tx_audio.kbps()
            header = (f"TX per stream: video {tx_v:.1f} kbps + audio {tx_a:.1f} kbps    "
                      f"(H.264 at {self.args.width}x{self.args.height} would need "
                      f"~{ref_kbps:.0f} kbps)")
            while not self._notices.empty():
                win.toast(self._notices.get_nowait())
            return ui.CallState(tiles, header, self._controls())

        def on_action(action: str):
            if action == "leave":
                self.running = False
                win.close()
            elif action == "recalibrate":
                self._calibrate_next = True
                self.encoder.reset_tracking()
                for r in self.remotes.values():
                    if r.renderer is not None:
                        r.renderer.reset_reference()
                if self.self_renderer is not None:
                    # Its calibrate packet may be overwritten in the one-deep
                    # slot before it renders, so reset directly as well.
                    self.self_renderer.reset_reference()
                win.toast("Re-calibrating neutral pose")
                print("Re-calibrating neutral pose.")
            elif action == "preview":
                self.set_self_preview(not self.self_preview)
            elif action == "translate":
                self.cycle_translation()
            elif action == "decode_every":
                self.cycle_decode_every()
            elif action == "selfview":
                self.self_view = "camera" if self.self_view == "avatar" else "avatar"
                win.toast("Showing your camera" if self.self_view == "camera"
                          else "Showing your generated avatar")
            elif action == "snapshot":
                path = f"conference_{int(time.time())}.png"
                if win.snapshot(path):
                    win.toast(f"Saved {path}")
                    print(f"Saved {path}")
                else:
                    win.toast("Could not save the snapshot")

        win = ui.CallWindow(WINDOW_TITLE, self.args.room, self.args.tile, refresh, on_action)
        win.toast(f"Joined '{self.args.room}' — press R while facing the camera "
                  "to calibrate", 4.0)
        ui.exec_window(win)

    def _controls(self) -> list:
        on = self.self_preview
        label = "Preparing…" if self._self_building else f"Self preview: {'on' if on else 'off'}"
        toggle = ui.Control("P", label, "preview", "toggle_on" if on else "default")
        controls = [toggle] + ([ui.SELF_VIEW_CONTROL] if on else [])
        if self.speaker is not None:
            if self._translate_loading:
                label = f"Preparing {LANGUAGES[self._translate_loading].name}…"
            elif self.translate_to:
                label = f"Translate: {LANGUAGES[self.translate_to].name}"
            else:
                label = "Translate: off"
            controls.append(ui.Control("T", label, "translate",
                                       "toggle_on" if self.translate_to else "default"))
        render_label = ("Render: full rate" if self.decode_every == 1
                       else f"Render: 1/{self.decode_every} rate")
        controls.append(ui.Control("D", render_label, "decode_every",
                                   "toggle_on" if self.decode_every > 1 else "default"))
        return controls + ui.CALL_CONTROLS

    def _self_tile(self) -> ui.Tile:
        slot = self.self_slot.peek()
        status, camera = slot if slot else ("waiting", None)
        renderer = self.self_renderer if self.self_preview else None
        generated = None
        if renderer is not None:
            generated = renderer.last_frame_bgr if self._self_frames else renderer.source_bgr_256

        # Derived from the same meter as the bitrate, so the two always agree —
        # frames dropped for a missing face show up in both.
        subtitle = (f"{self._tx_video.pps():.1f} fps sent   "
                    f"extract {self._extract_ms:.0f} ms   "
                    f"{P.MOTION_PACKET_SIZE} B/frame")
        if renderer is not None:
            subtitle += f"   self render {self._self_render_ms:.0f} ms"

        if self.self_view == "avatar" and generated is not None:
            image, inset = generated, camera
            badge, inset_label = "Generated · what others see", "camera"
        else:
            image, inset = camera, generated
            badge, inset_label = ("Camera" if renderer is not None
                                  else "Camera · self preview off"), "generated"
        if image is None:
            image, inset = inset, None
        badge_color = ui.ACCENT
        if status == "noface":
            badge, badge_color = "No face detected · not sending", ui.RED

        return ui.Tile(
            title=f"{self.args.name} (you)",
            image=image,
            subtitle=subtitle,
            status="starting camera…",
            accent=ui.GREEN if status == "ok" else ui.RED,
            inset=inset,
            inset_label=inset_label,
            badge=badge if image is not None else "",
            badge_color=badge_color,
        )

    def _remote_tile(self, remote: RemoteState) -> ui.Tile:
        peer = self.peers.get(remote.peer_id)
        image = remote.renderer.last_frame_bgr if remote.renderer is not None else None
        if remote.renderer is not None and remote.frames_rendered == 0:
            image = remote.renderer.source_bgr_256

        if peer is not None:
            v_kbps, a_kbps = peer.stats_rx.kbps()
            have_sync = remote.av_sync.playhead_t_ms() is not None
            sync = f"{remote.sync_drift_ms:+.0f} ms" if have_sync else "n/a"
            subtitle = (f"video {v_kbps:.1f} kbps   audio {a_kbps:.1f} kbps   "
                        f"render {remote.render_ms:.0f} ms   lost {remote.lost}   "
                        f"a/v sync {sync}")
            if self.translate_to and remote.translation is not None:
                subtitle += (f"   → {self.translate_to} "
                             f"lag {remote.translation.lag_ms / 1000:.1f} s")
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
        finally:
            self.running = False
            thread.join(timeout=5.0)
            if self.camera is not None:
                self.camera.close()
            if self.mic is not None:
                self.mic.close()
            if self.speaker is not None:
                self.speaker.close()
            print("Left the room.")


def _ice_servers(args):
    """STUN URLs plus any TURN relay, which needs credentials alongside its URL.

    On a network with client isolation — most campus and guest wifi — neither
    host nor server-reflexive candidates are reachable and the call only works
    through a relay, so TURN is not an exotic fallback there but the only path.
    """
    servers = [RTCIceServer(urls=u) for u in (args.stun or [])]
    for url in (args.turn or []):
        if not args.turn_user or not args.turn_pass:
            log.warning("TURN server %s given without --turn-user/--turn-pass; "
                        "almost every relay rejects anonymous allocations", url)
        servers.append(RTCIceServer(urls=url, username=args.turn_user,
                                    credential=args.turn_pass))
    return build_ice_servers(servers)


def _describe_ice(servers) -> str:
    if not servers:
        return "none (host candidates only — same-subnet, no isolation)"
    parts = []
    for s in servers:
        urls = s.urls if isinstance(s.urls, str) else ", ".join(s.urls)
        parts.append(f"{urls}{' (auth)' if s.username else ''}")
    return "; ".join(parts)


def _swallow(fn, *a):
    try:
        fn(*a)
    except Exception as exc:
        log.debug("send failed: %s", exc)


def _default_name() -> str:
    return os.environ.get("USERNAME") or os.environ.get("USER") or "guest"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Neural web conference: LivePortrait motion vectors + EnCodec over aiortc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--avatar", "-a",
                   help="Portrait image that represents you (omit to open the lobby)")
    p.add_argument("--name", "-n", default=None,
                   help="Display name (default: last one used, else $USERNAME)")
    p.add_argument("--room", "-r", default=None, help="Room to join (default: demo)")
    p.add_argument("--signaling", default=None,
                   help=f"Signaling websocket URL (default: {DEFAULT_SIGNALING})")
    p.add_argument("--lobby", action="store_true",
                   help="Always show the lobby, even when --avatar is given")
    p.add_argument("--stun", nargs="*", default=[DEFAULT_STUN],
                   help="STUN URLs (pass with no values to disable)")
    p.add_argument("--turn", nargs="*", default=[],
                   help="TURN relay URLs, e.g. turn:host:3478 or turns:host:5349 "
                        "(needed on networks with client isolation)")
    p.add_argument("--turn-user", default=os.environ.get("TURN_USER"),
                   help="TURN username (or set TURN_USER)")
    p.add_argument("--turn-pass", default=os.environ.get("TURN_PASS"),
                   help="TURN password (or set TURN_PASS)")

    g = p.add_argument_group("video")
    g.add_argument("--camera", "-c", type=int, default=0,
                   help="Camera index, or -1 to run without one (no video sent or "
                        "shown for you; useful for a second, listen-only client on "
                        "a machine with only one camera)")
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
    g.add_argument("--decode-every", type=int, default=1, choices=DECODE_EVERY_LEVELS,
                   help="Generate a new face only every N due motion packets per tile "
                        "(self-preview included); skipped turns keep showing the last "
                        "rendered frame. Frees GPU time for the rest of the room at the "
                        "cost of that tile's own update rate. Cycle with D in the call.")
    g.add_argument("--preview-self", action=argparse.BooleanOptionalAction, default=None,
                   help="Render your own avatar locally so you see what others see "
                        "(costs one extra warp+decode per frame; toggle with P in the "
                        "call; default: on, or your last lobby choice)")
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

    g = p.add_argument_group("translation (English speech, translated on the receiving side)")
    g.add_argument("--translate", choices=sorted(LANGUAGES), default=None,
                   help="Start the call hearing everyone in this language")
    g.add_argument("--translate-langs", nargs="+", choices=sorted(LANGUAGES),
                   default=["es", "fr"],
                   help="Languages the T key cycles through, after off")
    g.add_argument("--asr-model", default="small",
                   help="faster-whisper model: tiny, base, small, small.en, medium, …")
    g.add_argument("--mt-model", choices=["base", "big"], default="base",
                   help="Opus-MT size; `big` (transformer-big, es/fr/it) is ~3x the "
                        "download for a modest gain")
    g.add_argument("--asr-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Where Whisper runs; the GPU cuts ~1.5 s of lag (default: GPU "
                        "if there is one, unless --cpu)")
    g.add_argument("--translate-threads", type=int, default=4,
                   help="CPU threads for speech recognition")
    g.add_argument("--tts", choices=["piper", "pocket"], default="piper",
                   help="Speech synthesis engine for the translated voice: `piper` "
                        "(one fixed voice per language) or `pocket` (Kyutai's "
                        "Pocket TTS, voice-cloned from --voice-reference, runs on CPU)")
    g.add_argument("--voice-reference", default=None,
                   help="Wav file, named Pocket TTS voice (e.g. alba), or "
                        "hf://kyutai/tts-voices/... path to clone — `--tts pocket` only "
                        f"(default: the bundled voice '{DEFAULT_POCKET_VOICE}')")
    g.add_argument("--pocket-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="`--tts pocket` only; auto stays on CPU (GPU use is unofficial)")
    g.add_argument("--pocket-quantize", action="store_true",
                   help="`--tts pocket` only; int8 dynamic quantization on CPU "
                        "(smaller/faster, same quality)")

    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.cpu and args.asr_device == "auto":
        args.asr_device = "cpu"
    if args.translate and args.translate not in args.translate_langs:
        args.translate_langs.insert(0, args.translate)
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
    ui.ensure_app()
    if args.lobby or not args.avatar:
        prefs = ui.load_prefs()
        result = ui.run_lobby(
            name=args.name or prefs.get("name") or _default_name(),
            room=args.room or "",
            avatar=args.avatar or prefs.get("avatar"),
            signaling=args.signaling or prefs.get("signaling") or DEFAULT_SIGNALING,
            camera=args.camera,
            examples_dir=EXAMPLES_DIR,
            preview_self=(args.preview_self if args.preview_self is not None
                          else prefs.get("preview_self", True)))
        if result is None:
            print("Lobby closed; not joining.")
            return 0
        args.name, args.room = result.name, result.room
        args.avatar, args.signaling = result.avatar, result.signaling
        args.preview_self = result.preview_self
        ui.save_prefs(name=result.name, avatar=result.avatar,
                      signaling=result.signaling, room=result.room,
                      preview_self=result.preview_self)
        print(f"{'Creating' if result.created else 'Joining'} room '{args.room}' "
              f"as {args.name}")

    args.name = args.name or _default_name()
    args.room = args.room or "demo"
    args.signaling = args.signaling or DEFAULT_SIGNALING
    if args.preview_self is None:
        args.preview_self = True
    if not os.path.exists(args.avatar):
        print(f"error: avatar not found: {args.avatar}", file=sys.stderr)
        return 2

    # Model loading takes seconds; it runs on a worker thread behind a loading
    # window so the GUI stays responsive (and closable) meanwhile.
    try:
        app = ui.run_with_loading(WINDOW_TITLE, "Loading LivePortrait…",
                                  f"Joining '{args.room}' as {args.name}",
                                  lambda: ConferenceApp(args))
    except KeyboardInterrupt:
        print("Cancelled while loading.")
        return 130
    except Exception as exc:
        log.exception("could not start the call")
        ui.show_error("Could not start the call", str(exc))
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
