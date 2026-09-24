# coding: utf-8
"""Local capture and playback devices, plus the playout timing built on them.

All of it is non-blocking with respect to callers. The A/V sync pieces live
here rather than in the client because they are defined by the speaker: it is
SpeakerMixer's drain rate that says when a sample becomes audible, and that is
the clock everything else is scheduled against.
"""

import queue
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from . import protocol as P

try:
    import sounddevice as sd
except Exception:                                   # pragma: no cover
    sd = None


class CameraReader:
    """Reads the webcam on its own thread and keeps only the newest frame."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if _is_windows() else 0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class MicReader:
    """Delivers fixed-size mono float32 chunks to a callback on the audio thread."""

    def __init__(self, on_chunk: Callable[[np.ndarray], None],
                 chunk_samples: int, device: Optional[int] = None,
                 sample_rate: Optional[int] = None):
        if sd is None:
            raise RuntimeError("sounddevice is not available")
        info = sd.query_devices(device, "input")
        self.sample_rate = int(sample_rate or info["default_samplerate"])
        self.name = info["name"]
        self._on_chunk = on_chunk
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=chunk_samples, device=device, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        self._on_chunk(indata[:, 0].copy())

    def start(self):
        self._stream.start()

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class SpeakerMixer:
    """Sums every peer's decoded audio into one output stream.

    Each peer gets its own ring buffer so a slow or bursty peer stalls only
    itself; the callback mixes whatever is available and pads with silence.
    """

    def __init__(self, sample_rate: int = 24000, blocksize: int = 480,
                 device: Optional[int] = None, buffer_seconds: float = 1.5):
        if sd is None:
            raise RuntimeError("sounddevice is not available")
        self.sample_rate = sample_rate
        self._cap = int(sample_rate * buffer_seconds)
        self._buffers: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.underruns = 0
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            blocksize=blocksize, device=device, callback=self._cb)

    def _cb(self, outdata, frames, time_info, status):
        mix = np.zeros(frames, dtype=np.float32)
        starved = False
        with self._lock:
            for pid, buf in list(self._buffers.items()):
                if buf.size == 0:
                    continue
                n = min(frames, buf.size)
                mix[:n] += buf[:n]
                self._buffers[pid] = buf[n:]
                if n < frames:
                    starved = True
        if starved:
            self.underruns += 1
        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:, 0] = mix

    def push(self, peer_id: str, pcm: np.ndarray):
        with self._lock:
            buf = self._buffers.get(peer_id)
            buf = pcm.astype(np.float32) if buf is None else np.concatenate([buf, pcm])
            if buf.size > self._cap:               # late audio is useless audio
                buf = buf[-self._cap:]
            self._buffers[peer_id] = buf

    def drop(self, peer_id: str):
        with self._lock:
            self._buffers.pop(peer_id, None)

    def backlog_ms(self, peer_id: str) -> float:
        with self._lock:
            buf = self._buffers.get(peer_id)
        return 0.0 if buf is None else buf.size / self.sample_rate * 1000

    @property
    def output_latency_ms(self) -> float:
        """What the device holds behind our own buffer. MME and shared-mode
        WASAPI report tens of milliseconds here, which is the difference
        between lip sync landing and being visibly early."""
        try:
            return float(self._stream.latency) * 1000
        except Exception:
            return 0.0

    def start(self):
        self._stream.start()

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class LatestSlot:
    """A one-deep mailbox: writers overwrite, readers take. Never blocks a writer."""

    def __init__(self):
        self._item = None
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._item = item

    def take(self):
        with self._lock:
            item, self._item = self._item, None
            return item

    def peek(self):
        with self._lock:
            return self._item


class AVSync:
    """Tracks where in a remote's capture timeline their audio is playing.

    Motion and audio packets from a given peer are both stamped with that
    peer's `time.monotonic()` clock, so the two are directly comparable even
    though we never negotiate clocks with them. The speaker drains each
    peer's ring buffer at a fixed sample rate, so whatever is queued for that
    peer is a countdown to when the chunk we are about to add actually
    becomes audible; anchoring the chunk's capture timestamp to that moment
    turns the pair into a live estimate of which capture instant is coming
    out of the speaker right now.

    Re-anchoring on every chunk is what keeps this honest: it costs nothing
    and it absorbs codec padding, mixer underruns and clock drift between
    the two machines, none of which can then accumulate.
    """

    def __init__(self):
        # One tuple, so the render and UI threads can never read a new
        # timestamp against a stale anchor.
        self._anchor: Optional[Tuple[int, float]] = None

    def note_audio_push(self, t_ms: int, delay_ms: float) -> None:
        """Audio captured at `t_ms` becomes audible `delay_ms` from now."""
        self._anchor = (t_ms, time.perf_counter() + delay_ms / 1000.0)

    def playhead_t_ms(self) -> Optional[float]:
        """Sender-clock instant the speaker is presenting right now, or None
        before any audio has been played for this peer."""
        anchor = self._anchor
        if anchor is None:
            return None
        t_ms, audible_at = anchor
        return t_ms + (time.perf_counter() - audible_at) * 1000.0


class MotionJitterBuffer:
    """Motion packets waiting for their moment on the peer's audio playhead.

    Video has to queue here rather than sit in a one-deep mailbox. Motion
    arrives well ahead of the audio it belongs with — the sender's voice is
    still being gathered into a chunk and encoded while the pose that made it
    is already on the wire — so keeping only the newest packet and rendering
    it once due would never render at all: the wait is longer than the gap
    between packets, so each one is replaced before its turn comes. Queueing
    them lets the buffer fill to whatever head start the audio path has and
    then drain at capture cadence.

    Late packets are still dropped, just at the far end: when the playhead
    passes several at once only the newest of them is rendered, which keeps
    the old "never build a backlog on the GPU" behaviour intact.
    """

    def __init__(self, maxlen: int = 90):
        self._q: "deque[P.MotionPacket]" = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._released_t_ms: Optional[int] = None

    def put(self, pkt: P.MotionPacket) -> None:
        with self._lock:
            if (self._released_t_ms is not None
                    and P.diff_t_ms(pkt.t_ms, self._released_t_ms) <= 0):
                return          # unordered channel reordered it past its slot
            self._q.append(pkt)             # maxlen drops the oldest if we stall

    def take_due(self, playhead_t_ms: Optional[float],
                 slack_ms: float) -> Optional[P.MotionPacket]:
        """The newest packet the playhead has reached, or None while the whole
        queue is still in the peer's future. Without a playhead — a peer with
        no audio yet — the newest packet is always due."""
        with self._lock:
            due = None
            if playhead_t_ms is None:
                if self._q:
                    due = self._q[-1]
                self._q.clear()
            else:
                mark = int(playhead_t_ms)
                while self._q and P.diff_t_ms(self._q[0].t_ms, mark) <= slack_ms:
                    pkt = self._q.popleft()
                    if due is None or P.diff_t_ms(pkt.t_ms, due.t_ms) > 0:
                        due = pkt
            if due is not None:
                self._released_t_ms = due.t_ms
            return due


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")


def probe_input_rate(device: Optional[int] = None) -> int:
    """The device's native capture rate, without opening a stream."""
    if sd is None:
        raise RuntimeError("sounddevice is not available")
    return int(sd.query_devices(device, "input")["default_samplerate"])


def list_devices() -> str:
    if sd is None:
        return "sounddevice unavailable"
    return str(sd.query_devices())
