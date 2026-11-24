# coding: utf-8
"""
Opus voice codec for the conference audio path: 24 kHz mono, 20 ms packets.

Uses `pyogg.opus` (what the conference app already uses in realtime_mic.py), or
the system libopus through ctypes when PyOgg is not installed:

    pip install PyOgg                      # bundles libopus on Windows
    sudo apt install libopus0              # Linux / WSL2

This file is identical on the server (translation_server/src/opus.py) and in the
conference app (webrtc_conference/opus_codec.py); keep them in step.
"""

import ctypes
import ctypes.util
from typing import Optional

import numpy as np

SAMPLE_RATE = 24000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000       # 480
MAX_PACKET = 1500
_MAX_DECODE = SAMPLE_RATE * 120 // 1000              # longest Opus frame (120 ms)

_APPLICATION_VOIP = 2048
_SET_BITRATE = 4002


class _RawOpus:
    """The handful of libopus calls used here, bound by hand (no PyOgg)."""

    OPUS_APPLICATION_VOIP = _APPLICATION_VOIP
    OPUS_SET_BITRATE_REQUEST = _SET_BITRATE

    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        c = ctypes
        lib.opus_encoder_create.restype = c.c_void_p
        lib.opus_encoder_create.argtypes = [c.c_int32, c.c_int, c.c_int, c.POINTER(c.c_int)]
        lib.opus_decoder_create.restype = c.c_void_p
        lib.opus_decoder_create.argtypes = [c.c_int32, c.c_int, c.POINTER(c.c_int)]
        lib.opus_encode.restype = c.c_int32
        lib.opus_encode.argtypes = [c.c_void_p, c.POINTER(c.c_int16), c.c_int,
                                    c.POINTER(c.c_ubyte), c.c_int32]
        lib.opus_decode.restype = c.c_int
        lib.opus_decode.argtypes = [c.c_void_p, c.POINTER(c.c_ubyte), c.c_int32,
                                    c.POINTER(c.c_int16), c.c_int, c.c_int]
        lib.opus_encoder_destroy.argtypes = [c.c_void_p]
        lib.opus_decoder_destroy.argtypes = [c.c_void_p]
        self._lib = lib
        self.opus_encoder_create = lib.opus_encoder_create
        self.opus_decoder_create = lib.opus_decoder_create
        self.opus_encode = lib.opus_encode
        self.opus_decode = lib.opus_decode
        self.opus_encoder_ctl = lib.opus_encoder_ctl          # variadic: no argtypes
        self.opus_encoder_destroy = lib.opus_encoder_destroy
        self.opus_decoder_destroy = lib.opus_decoder_destroy


_lib = None


def _opus():
    global _lib
    if _lib is not None:
        return _lib
    try:
        import pyogg.opus as lib
        _lib = lib
        return _lib
    except Exception as pyogg_error:
        path = ctypes.util.find_library("opus")
        if path is None:
            raise RuntimeError(
                "Opus is not available: `pip install PyOgg` (Windows) or "
                "`sudo apt install libopus0` (Linux)") from pyogg_error
        _lib = _RawOpus(path)
        return _lib


class OpusEncoder:
    """float32 mono, exactly FRAME_SAMPLES per call -> one Opus packet."""

    def __init__(self, bitrate_kbps: float = 24.0):
        self._op = op = _opus()
        err = ctypes.c_int(0)
        application = getattr(op, "OPUS_APPLICATION_VOIP", _APPLICATION_VOIP)
        self._enc = op.opus_encoder_create(SAMPLE_RATE, 1, application, ctypes.byref(err))
        if err.value != 0 or not self._enc:
            raise RuntimeError(f"Opus encoder creation failed ({err.value})")
        op.opus_encoder_ctl(self._enc, getattr(op, "OPUS_SET_BITRATE_REQUEST", _SET_BITRATE),
                            ctypes.c_int(int(bitrate_kbps * 1000)))
        self._buf = (ctypes.c_ubyte * MAX_PACKET)()

    def encode(self, pcm: np.ndarray) -> bytes:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size != FRAME_SAMPLES:
            raise ValueError(f"need {FRAME_SAMPLES} samples per Opus frame, got {pcm.size}")
        s16 = np.ascontiguousarray(np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16))
        ptr = s16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        n = self._op.opus_encode(self._enc, ptr, FRAME_SAMPLES, self._buf, MAX_PACKET)
        if n < 0:
            raise RuntimeError(f"opus_encode failed ({n})")
        return bytes(bytearray(self._buf[:n]))


class OpusDecoder:
    """One Opus packet -> float32 mono at 24 kHz. Keeps decoder state, so use
    one instance per incoming stream."""

    def __init__(self):
        self._op = op = _opus()
        err = ctypes.c_int(0)
        self._dec = op.opus_decoder_create(SAMPLE_RATE, 1, ctypes.byref(err))
        if err.value != 0 or not self._dec:
            raise RuntimeError(f"Opus decoder creation failed ({err.value})")
        self._out = np.zeros(_MAX_DECODE, dtype=np.int16)

    def decode(self, payload: bytes) -> np.ndarray:
        n = len(payload)
        if n == 0:
            return np.zeros(0, np.float32)
        data = (ctypes.c_ubyte * n).from_buffer_copy(payload)
        ptr = self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        got = self._op.opus_decode(self._dec, data, n, ptr, _MAX_DECODE, 0)
        if got < 0:
            raise ValueError(f"bad Opus packet ({got})")
        return self._out[:got].astype(np.float32) / 32768.0


def peek_available() -> Optional[str]:
    """None when Opus can be used, else the reason it can't."""
    try:
        _opus()
        return None
    except Exception as exc:
        return str(exc)
