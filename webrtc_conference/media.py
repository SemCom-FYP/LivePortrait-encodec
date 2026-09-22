# coding: utf-8
"""Local capture and playback devices, all non-blocking with respect to callers."""

import queue
import threading
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

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
