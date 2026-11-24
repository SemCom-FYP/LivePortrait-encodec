# coding: utf-8
"""
The conference's audio path: Opus through the translation server.

    mic ─▶ resample 24 kHz ─▶ 20 ms Opus packets ─▶ server ─┬▶ forwarded untouched
                                                            └▶ or translated first
    server ─▶ Opus packets from every other participant ─▶ decode ─▶ on_voice(...)

Nothing about audio travels over the WebRTC datachannels any more; those carry
the motion vectors and the portrait only. What keeps lips on the voice is the
`t_ms` every packet carries, the sender's capture clock — the same clock the
motion packets are stamped with. The server preserves it: a pass-through packet
keeps the sender's own stamp, and a translated packet is stamped with when its
source speech began plus the offset into the translation. The receiver feeds it
to AVSync exactly as before, so video renders against the audible voice.

Transport: the WebSocket to the server always carries control (join, language
changes, WebRTC signaling). Audio starts on it too, then moves to an unreliable
WebRTC datachannel to the server (UDP: a late packet is dropped rather than
stalling everything behind it, as TCP would). If the channel can't be set up —
server without aiortc, UDP blocked — audio simply stays on the WebSocket.

The relay reconnects by itself if the server drops, and the call carries on
(video keeps flowing) while it is away.
"""

import asyncio
import json
import logging
import struct
import threading
from typing import Callable, Optional
from urllib.parse import urlencode

import numpy as np
from aiohttp import WSMsgType

from . import protocol as P
from .opus_codec import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, OpusDecoder, OpusEncoder

log = logging.getLogger("audio_relay")

MAGIC_UP = ord("U")
MAGIC_DOWN = ord("V")
VERSION = 1
FLAG_TRANSLATED = 1 << 0
_UP = struct.Struct("<BBHI")                # magic, version, seq, t_ms
_DOWN = struct.Struct("<BBBBHI")            # magic, version, flags, id_len, seq, t_ms
_MASK = 0xFFFFFFFF
_REANCHOR_MS = 30                           # mic clock vs wall clock drift tolerated

DEFAULT_SERVER = "ws://10.10.1.147:8770/v1/conf"
RTC_CHANNEL_ID = 1                          # pre-negotiated "voice" channel, both ends
RTC_PACKET_LIFETIME_MS = 300
RTC_BACKPRESSURE_BYTES = 32 * 1024

# Names for the T key and the HUD. The server can speak 36 languages; these are
# the common ones, and any other code can still be passed with --translate.
LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian",
    "sv": "Swedish", "vi": "Vietnamese", "id": "Indonesian", "th": "Thai",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


class _MicResampler:
    """Native mic rate -> 24 kHz, carrying filter state across chunks."""

    def __init__(self, src_rate: int):
        self.src_rate = int(src_rate)
        self._stream = None
        self._torch = None
        if self.src_rate == SAMPLE_RATE:
            return
        try:
            import soxr
            self._stream = soxr.ResampleStream(self.src_rate, SAMPLE_RATE, 1,
                                               dtype="float32", quality="HQ")
        except Exception:
            import torchaudio.functional as AF          # per-chunk: slight edge artefacts
            self._torch = AF
            log.warning("soxr not installed; resampling the mic per chunk "
                        "(pip install soxr for cleaner audio)")

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        if self.src_rate == SAMPLE_RATE:
            return pcm
        if self._stream is not None:
            return self._stream.resample_chunk(pcm)
        import torch
        out = self._torch.resample(torch.from_numpy(pcm)[None, :], self.src_rate, SAMPLE_RATE)
        return out.squeeze(0).numpy()


class AudioRelay:
    """One participant's audio connection to the translation server."""

    def __init__(self, url: str, room: str, name: str, *, speaks: str = "en",
                 hear: Optional[str] = None, token: Optional[str] = None,
                 bitrate_kbps: float = 24.0, transport: str = "rtc",
                 ice_servers: Optional[list] = None,
                 on_voice: Callable[[str, int, np.ndarray, bool], None],
                 on_tx: Optional[Callable[[int], None]] = None,
                 on_rx: Optional[Callable[[str, int], None]] = None,
                 on_notice: Optional[Callable[[str], None]] = None):
        self.url, self.room, self.name = url, room, name
        self.speaks, self.hear, self.token = speaks, hear, token
        self.on_voice, self.on_tx, self.on_rx = on_voice, on_tx, on_rx
        self.on_notice = on_notice or (lambda msg: None)
        self.transport = transport                      # "rtc" (WebRTC, WebSocket fallback) | "ws"
        self.ice_servers = ice_servers
        self.mic_rate = SAMPLE_RATE
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.peer_id: Optional[str] = None
        self.connected = threading.Event()
        self._enc = OpusEncoder(bitrate_kbps)           # raises early if Opus is missing
        self._decoders: dict = {}
        self._resampler: Optional[_MicResampler] = None
        self._buf = np.zeros(0, np.float32)
        self._buf_t = 0.0
        self._seq = 0
        self._q: Optional[asyncio.Queue] = None
        self._closing = False
        self._ws = None
        self._pc = None                                 # aiortc connection to the server
        self._dc = None                                 # its "voice" channel, once open
        self._rtc_notified = False
        self._rtc_task = None

    # ── lifecycle (event loop thread) ────────────────────────────────────────

    def attach(self, loop: asyncio.AbstractEventLoop, peer_id: str):
        """Called once the signaling server has given us our id."""
        self.loop, self.peer_id = loop, peer_id
        self._q = asyncio.Queue(maxsize=100)

    async def run(self):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            failures = 0
            while not self._closing:
                try:
                    async with session.ws_connect(self._url(), heartbeat=20,
                                                  max_msg_size=0) as ws:
                        failures = 0
                        await self._serve(ws)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if failures == 0:
                        self.on_notice(f"Voice server unreachable ({exc}); retrying…")
                    log.warning("voice server: %s", exc)
                    failures += 1
                finally:
                    self.connected.clear()
                    self._ws = None
                if not self._closing:
                    await asyncio.sleep(3.0)

    async def _serve(self, ws):
        self._ws = ws
        self._reset_tx()
        self.connected.set()
        sender = asyncio.create_task(self._send_loop(ws))
        if self.transport == "rtc":
            self._rtc_task = asyncio.create_task(self._rtc_start(ws))   # keep a reference
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    self._on_down(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await self._on_text(msg.data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING,
                                  WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        finally:
            sender.cancel()
            await self._rtc_close()
        if not self._closing:
            self.on_notice("Voice server disconnected; reconnecting…")

    # ── WebRTC upgrade of the audio path ─────────────────────────────────────

    async def _rtc_start(self, ws):
        """Offer the server a WebRTC connection; audio moves to it once the
        datachannel opens. Any failure just leaves audio on the WebSocket."""
        try:
            from aiortc import RTCConfiguration, RTCPeerConnection
        except ImportError:
            log.warning("aiortc is not installed; voice stays on the WebSocket")
            return
        await self._rtc_close()
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=self.ice_servers or []))
        dc = pc.createDataChannel("voice", negotiated=True, id=RTC_CHANNEL_ID, ordered=True,
                                  maxPacketLifeTime=RTC_PACKET_LIFETIME_MS)
        self._pc = pc

        @dc.on("open")
        def _open():
            self._dc = dc
            log.info("voice moved to a WebRTC datachannel")
            if not self._rtc_notified:
                self._rtc_notified = True
                self.on_notice("Voice: WebRTC")

        @dc.on("close")
        def _closed():
            if self._dc is dc:
                self._dc = None

        @dc.on("message")
        def _message(data):
            if isinstance(data, bytes):
                self._on_down(data)

        @pc.on("connectionstatechange")
        async def _state():
            if pc.connectionState in ("failed", "closed") and self._dc is dc:
                self._dc = None
                log.info("WebRTC %s; voice back on the WebSocket", pc.connectionState)

        try:
            await pc.setLocalDescription(await pc.createOffer())
            await ws.send_str(json.dumps({"type": "rtc_offer", "sdp": pc.localDescription.sdp}))
        except Exception as exc:
            log.warning("could not start WebRTC voice (%s); using the WebSocket", exc)
            await self._rtc_close()

    async def _rtc_answer(self, sdp: str):
        from aiortc import RTCSessionDescription
        if self._pc is None:
            return
        try:
            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
        except Exception as exc:
            log.warning("WebRTC answer rejected (%s); using the WebSocket", exc)
            await self._rtc_close()

    async def _rtc_close(self):
        pc, self._pc, self._dc = self._pc, None, None
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass

    async def close(self):
        self._closing = True
        ws = self._ws
        if ws is not None and not ws.closed:
            try:
                await ws.send_str(json.dumps({"type": "leave"}))
                await ws.close()
            except Exception:
                pass

    def _url(self) -> str:
        query = {"room": self.room, "peer": self.peer_id, "name": self.name,
                 "speaks": self.speaks or ""}
        if self.hear:
            query["hear"] = self.hear
        if self.token:
            query["token"] = self.token
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}{urlencode(query)}"

    async def _send_loop(self, ws):
        while True:
            item = await self._q.get()
            if isinstance(item, str):
                await ws.send_str(item)
            else:
                dc = self._dc
                if dc is not None and dc.readyState == "open":
                    if dc.bufferedAmount > RTC_BACKPRESSURE_BYTES:
                        continue                        # congested: drop, stay live
                    try:
                        dc.send(item)
                    except Exception:
                        self._dc = None
                        await ws.send_bytes(item)
                else:
                    await ws.send_bytes(item)
                if self.on_tx is not None:
                    self.on_tx(len(item))

    # ── what to hear ─────────────────────────────────────────────────────────

    def set_hear(self, lang: Optional[str]):
        """Hear everyone in `lang`, or untouched originals with None. Thread-safe."""
        self.hear = lang or None
        if self.loop is not None:
            msg = json.dumps({"type": "configure", "hear": self.hear})
            self.loop.call_soon_threadsafe(self._put, msg)

    # ── mic -> server (mic thread) ───────────────────────────────────────────

    def send_mic(self, pcm: np.ndarray, t_ms: int):
        """One mic chunk at `mic_rate`, `t_ms` being the sender-clock time of its
        first sample. Called on the audio thread: encodes and hands off, never blocks."""
        if not self.connected.is_set() or self.loop is None:
            return
        if self._resampler is None:
            self._resampler = _MicResampler(self.mic_rate)
        pcm24 = self._resampler(pcm)
        if pcm24.size == 0:
            return
        if self._buf.size == 0:
            self._buf_t = float(t_ms)
        else:
            expected = int(self._buf_t + self._buf.size * 1000.0 / SAMPLE_RATE) & _MASK
            drift = P.diff_t_ms(t_ms, expected)
            if abs(drift) > _REANCHOR_MS:               # the mic's clock slipped: re-anchor
                self._buf_t += drift
        self._buf = np.concatenate([self._buf, pcm24])
        while self._buf.size >= FRAME_SAMPLES:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            stamp = int(self._buf_t) & _MASK
            self._buf_t += FRAME_MS
            try:
                packet = self._enc.encode(frame)
            except Exception as exc:
                log.warning("opus encode failed: %s", exc)
                continue
            self._seq = (self._seq + 1) & 0xFFFF
            data = _UP.pack(MAGIC_UP, VERSION, self._seq, stamp) + packet
            self.loop.call_soon_threadsafe(self._put, data)

    def _put(self, item):
        q = self._q
        if q is None:
            return
        if q.full():
            try:
                q.get_nowait()                          # network behind: drop the oldest
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)

    def _reset_tx(self):
        self._buf = np.zeros(0, np.float32)
        while self._q is not None and not self._q.empty():
            self._q.get_nowait()

    # ── server -> speaker (event loop thread) ────────────────────────────────

    def _on_down(self, data: bytes):
        if len(data) <= _DOWN.size or data[0] != MAGIC_DOWN:
            return
        _, _, flags, id_len, _seq, t_ms = _DOWN.unpack_from(data)
        start = _DOWN.size + id_len
        peer_id = data[_DOWN.size:start].decode(errors="replace")
        translated = bool(flags & FLAG_TRANSLATED)
        key = (peer_id, translated)
        decoder = self._decoders.get(key)
        if decoder is None:
            # A fresh decoder per (speaker, original|translated): the two are
            # different streams and share no codec state.
            decoder = self._decoders[key] = OpusDecoder()
        try:
            pcm = decoder.decode(data[start:])
        except ValueError as exc:
            log.debug("bad voice packet from %s: %s", peer_id, exc)
            return
        if self.on_rx is not None:
            self.on_rx(peer_id, len(data))
        if pcm.size:
            self.on_voice(peer_id, t_ms, pcm, translated)

    async def _on_text(self, text: str):
        try:
            ev = json.loads(text)
        except ValueError:
            return
        kind = ev.get("type")
        if kind == "rtc_answer":
            await self._rtc_answer(ev.get("sdp", ""))
        elif kind == "error" and ev.get("code") in ("rtc_unavailable", "rtc_failed"):
            log.warning("voice server: %s", ev.get("message"))     # not worth a toast
            await self._rtc_close()
        elif kind == "error":
            self.on_notice(f"Voice server: {ev.get('message', ev.get('code'))}")
        elif kind == "utterance_end":
            log.info("[%s] heard: %s", ev.get("speaker"), ev.get("target_text"))
        elif kind == "joined":
            log.info("voice relay joined room %r", ev.get("room"))
