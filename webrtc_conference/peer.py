# coding: utf-8
"""
One WebRTC connection to one remote participant.

No RTP media sections are negotiated at all — everything rides four out-of-band
(pre-negotiated) SCTP DataChannels with fixed stream ids, so both ends agree on
the layout without DCEP round-trips:

  id 1  ctrl    reliable, ordered      JSON control messages
  id 2  blob    reliable, ordered      chunked avatar JPEG
  id 3  motion  unreliable, unordered  150-byte motion vectors, drop on loss
  id 4  audio   time-bounded, ordered  EnCodec chunks, useless once late

Motion and audio are deliberately lossy: retransmitting a 40 ms-old frame costs
more than skipping it.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from aiortc import (RTCConfiguration, RTCIceCandidate, RTCIceServer,
                    RTCPeerConnection, RTCSessionDescription)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

from . import protocol as P

log = logging.getLogger("peer")

CH_CTRL, CH_BLOB, CH_MOTION, CH_AUDIO = 1, 2, 3, 4
AUDIO_LIFETIME_MS = 300
MOTION_BACKPRESSURE_BYTES = 64 * 1024


class RateMeter:
    """Sliding-window byte/packet rate, so the HUD shows now rather than
    an average dragged down by model loading at startup."""

    def __init__(self, window: float = 3.0):
        self.window = window
        self._events: deque[tuple[float, int]] = deque()
        self.total_bytes = 0
        self.total_packets = 0

    def add(self, nbytes: int):
        now = time.monotonic()
        self._events.append((now, nbytes))
        self.total_bytes += nbytes
        self.total_packets += 1
        self._trim(now)

    def _trim(self, now: float):
        cutoff = now - self.window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def kbps(self) -> float:
        now = time.monotonic()
        self._trim(now)
        if not self._events:
            return 0.0
        return sum(n for _, n in self._events) * 8 / self.window / 1000

    def pps(self) -> float:
        now = time.monotonic()
        self._trim(now)
        return len(self._events) / self.window


@dataclass
class LinkStats:
    motion: RateMeter = field(default_factory=RateMeter)
    audio: RateMeter = field(default_factory=RateMeter)

    def add_motion(self, n: int):
        self.motion.add(n)

    def add_audio(self, n: int):
        self.audio.add(n)

    def kbps(self) -> tuple[float, float]:
        return self.motion.kbps(), self.audio.kbps()

    def fps(self) -> float:
        return self.motion.pps()


class NeuralPeer:
    """Wraps an RTCPeerConnection plus its four channels for a single remote peer."""

    def __init__(self, peer_id: str, name: str, is_offerer: bool,
                 send_signal: Callable[[str, dict], Awaitable[None]],
                 ice_servers: Optional[list[str]] = None):
        self.peer_id = peer_id
        self.name = name
        self.is_offerer = is_offerer
        self._send_signal = send_signal
        self.stats_rx = LinkStats()
        self.stats_tx = LinkStats()
        self.connected = asyncio.Event()
        self.closed = False

        config = RTCConfiguration(iceServers=[
            RTCIceServer(urls=u) for u in (ice_servers or ["stun:stun.l.google.com:19302"])
        ])
        self.pc = RTCPeerConnection(configuration=config)

        self.ctrl = self.pc.createDataChannel(
            "ctrl", negotiated=True, id=CH_CTRL, ordered=True)
        self.blob = self.pc.createDataChannel(
            "blob", negotiated=True, id=CH_BLOB, ordered=True)
        self.motion = self.pc.createDataChannel(
            "motion", negotiated=True, id=CH_MOTION, ordered=False, maxRetransmits=0)
        self.audio = self.pc.createDataChannel(
            "audio", negotiated=True, id=CH_AUDIO, ordered=True,
            maxPacketLifeTime=AUDIO_LIFETIME_MS)

        # Injected by the application.
        self.on_motion: Callable[[str, P.MotionPacket], None] = lambda pid, pkt: None
        self.on_audio: Callable[[str, P.AudioPacket], None] = lambda pid, pkt: None
        self.on_avatar: Callable[[str, bytes], None] = lambda pid, data: None
        self.on_ctrl: Callable[[str, dict], None] = lambda pid, msg: None
        self.on_state: Callable[[str, str], None] = lambda pid, state: None

        self._reassembler = P.BlobReassembler()
        self._wire_channels()

        @self.pc.on("connectionstatechange")
        async def _on_state():
            state = self.pc.connectionState
            log.info("[%s] connection %s", self.name, state)
            self.on_state(self.peer_id, state)
            if state == "connected":
                self.connected.set()
            elif state in ("failed", "closed"):
                self.connected.clear()

    def _wire_channels(self):
        @self.ctrl.on("message")
        def _ctrl(message):
            if isinstance(message, str):
                try:
                    self.on_ctrl(self.peer_id, json.loads(message))
                except json.JSONDecodeError:
                    log.warning("[%s] bad ctrl message", self.name)

        @self.blob.on("message")
        def _blob(message):
            if not isinstance(message, bytes):
                return
            try:
                payload = self._reassembler.feed(message)
            except ValueError:
                return
            if payload is not None:
                self.on_avatar(self.peer_id, payload)

        @self.motion.on("message")
        def _motion(message):
            if not isinstance(message, bytes):
                return
            try:
                pkt = P.unpack_motion(message)
            except ValueError as exc:
                log.debug("[%s] dropping motion packet: %s", self.name, exc)
                return
            self.stats_rx.add_motion(len(message))
            self.on_motion(self.peer_id, pkt)

        @self.audio.on("message")
        def _audio(message):
            if not isinstance(message, bytes):
                return
            try:
                pkt = P.unpack_audio(message)
            except ValueError as exc:
                log.debug("[%s] dropping audio packet: %s", self.name, exc)
                return
            self.stats_rx.add_audio(len(message))
            self.on_audio(self.peer_id, pkt)

    # ── negotiation ──────────────────────────────────────────────────────────

    async def start_offer(self):
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self._send_signal(self.peer_id, {
            "kind": "sdp",
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
        })

    async def handle_signal(self, data: dict):
        kind = data.get("kind")
        if kind == "sdp":
            desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await self.pc.setRemoteDescription(desc)
            if desc.type == "offer":
                answer = await self.pc.createAnswer()
                await self.pc.setLocalDescription(answer)
                await self._send_signal(self.peer_id, {
                    "kind": "sdp",
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                })
        elif kind == "ice" and data.get("candidate"):
            candidate = candidate_from_sdp(data["candidate"].split(":", 1)[1])
            candidate.sdpMid = data.get("sdpMid")
            candidate.sdpMLineIndex = data.get("sdpMLineIndex")
            await self.pc.addIceCandidate(candidate)

    # ── sending ──────────────────────────────────────────────────────────────

    def send_ctrl(self, message: dict) -> bool:
        if self.ctrl.readyState != "open":
            return False
        self.ctrl.send(json.dumps(message))
        return True

    def send_avatar(self, blob_id: int, jpeg: bytes) -> bool:
        if self.blob.readyState != "open":
            return False
        for chunk in P.pack_blob(blob_id, jpeg):
            self.blob.send(chunk)
        return True

    def send_motion(self, payload: bytes) -> bool:
        if self.motion.readyState != "open":
            return False
        if self.motion.bufferedAmount > MOTION_BACKPRESSURE_BYTES:
            return False                     # congested: skip rather than queue
        self.motion.send(payload)
        self.stats_tx.add_motion(len(payload))
        return True

    def send_audio(self, payload: bytes) -> bool:
        if self.audio.readyState != "open":
            return False
        self.audio.send(payload)
        self.stats_tx.add_audio(len(payload))
        return True

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            await self.pc.close()
        except Exception:
            pass
