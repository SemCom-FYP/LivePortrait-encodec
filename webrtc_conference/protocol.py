# coding: utf-8
"""
Wire format for the neural conferencing datachannels.

Three payload families, each self-describing via a one-byte magic:

  'M' motion  — one LivePortrait motion vector (fixed 150 bytes)
  'A' audio   — one EnCodec chunk (variable, bit-packed 10-bit codes)
  'B' blob    — a chunk of a larger reliable transfer (the avatar JPEG)

All integers are little-endian. Motion fields are int16 fixed-point rather than
float16: the expression deltas live around 1e-2, where float16's relative
mantissa costs more precision than a fixed 6e-5 step.
"""

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

VERSION = 1

MAGIC_MOTION = ord("M")
MAGIC_AUDIO = ord("A")
MAGIC_BLOB = ord("B")

# ── Motion ────────────────────────────────────────────────────────────────────

N_KP = 21                       # LivePortrait implicit keypoints
FLAG_CALIBRATE = 1 << 0         # "treat this frame as the new neutral reference"

# Fixed-point scales. Chosen so each field's dynamic range fits int16 with room
# to spare; the reciprocal is the quantisation step.
Q_ANGLE = 128.0                 # degrees,  range ±256,  step 0.0078
Q_SCALE = 8192.0                # unitless, range ±4,    step 1.2e-4
Q_TRANS = 8192.0                # kp units, range ±4,    step 1.2e-4
Q_EXP = 16384.0                 # kp units, range ±2,    step 6.1e-5

_MOTION_HEADER = struct.Struct("<BBHIBB")   # magic, version, seq, t_ms, flags, n_kp
MOTION_HEADER_SIZE = _MOTION_HEADER.size    # 10
MOTION_VALUES = 3 + 1 + 3 + N_KP * 3        # pitch/yaw/roll, scale, t, exp = 70
MOTION_PACKET_SIZE = MOTION_HEADER_SIZE + MOTION_VALUES * 2   # 150 bytes


@dataclass
class MotionPacket:
    seq: int
    t_ms: int
    calibrate: bool
    pitch: float
    yaw: float
    roll: float
    scale: float
    t: np.ndarray               # (3,) float32
    exp: np.ndarray             # (N_KP, 3) float32

    @property
    def nbytes(self) -> int:
        return MOTION_PACKET_SIZE


def pack_motion(seq: int, t_ms: int, pitch: float, yaw: float, roll: float,
                scale: float, t: np.ndarray, exp: np.ndarray,
                calibrate: bool = False) -> bytes:
    """Serialise one motion vector. `exp` is (N_KP, 3) or (N_KP*3,)."""
    flags = FLAG_CALIBRATE if calibrate else 0
    header = _MOTION_HEADER.pack(
        MAGIC_MOTION, VERSION, seq & 0xFFFF, t_ms & 0xFFFFFFFF, flags, N_KP)

    q = np.empty(MOTION_VALUES, dtype=np.int16)
    q[0] = _clip16(pitch * Q_ANGLE)
    q[1] = _clip16(yaw * Q_ANGLE)
    q[2] = _clip16(roll * Q_ANGLE)
    q[3] = _clip16(scale * Q_SCALE)
    q[4:7] = _clip16(np.asarray(t, dtype=np.float32).reshape(3) * Q_TRANS)
    q[7:] = _clip16(np.asarray(exp, dtype=np.float32).reshape(-1) * Q_EXP)
    return header + q.tobytes()


def unpack_motion(data: bytes) -> MotionPacket:
    magic, version, seq, t_ms, flags, n_kp = _MOTION_HEADER.unpack_from(data, 0)
    if magic != MAGIC_MOTION:
        raise ValueError(f"not a motion packet (magic={magic:#x})")
    if version != VERSION:
        raise ValueError(f"unsupported motion version {version}")
    if n_kp != N_KP:
        raise ValueError(f"peer uses {n_kp} keypoints, we expect {N_KP}")

    q = np.frombuffer(data, dtype=np.int16, offset=MOTION_HEADER_SIZE,
                      count=MOTION_VALUES).astype(np.float32)
    return MotionPacket(
        seq=seq,
        t_ms=t_ms,
        calibrate=bool(flags & FLAG_CALIBRATE),
        pitch=float(q[0] / Q_ANGLE),
        yaw=float(q[1] / Q_ANGLE),
        roll=float(q[2] / Q_ANGLE),
        scale=float(q[3] / Q_SCALE),
        t=q[4:7] / Q_TRANS,
        exp=(q[7:] / Q_EXP).reshape(N_KP, 3),
    )


def _clip16(x):
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


# ── Audio ─────────────────────────────────────────────────────────────────────

_AUDIO_HEADER = struct.Struct("<BBHBH")     # magic, version, seq, n_q, n_frames
AUDIO_HEADER_SIZE = _AUDIO_HEADER.size      # 7
CODEBOOK_BITS = 10                          # EnCodec codebooks hold 1024 entries


@dataclass
class AudioPacket:
    seq: int
    codes: np.ndarray           # (n_q, n_frames) int64


def pack_audio(seq: int, codes: np.ndarray) -> bytes:
    """Serialise EnCodec codes (n_q, n_frames) as a 10-bit-per-entry bitstream."""
    n_q, n_frames = codes.shape
    header = _AUDIO_HEADER.pack(
        MAGIC_AUDIO, VERSION, seq & 0xFFFF, n_q, n_frames)
    # Transmit frame-major so a truncated packet still decodes a prefix of time.
    return header + _bitpack10(np.ascontiguousarray(codes.T).ravel())


def unpack_audio(data: bytes) -> AudioPacket:
    magic, version, seq, n_q, n_frames = _AUDIO_HEADER.unpack_from(data, 0)
    if magic != MAGIC_AUDIO:
        raise ValueError(f"not an audio packet (magic={magic:#x})")
    if version != VERSION:
        raise ValueError(f"unsupported audio version {version}")
    flat = _bitunpack10(data[AUDIO_HEADER_SIZE:], n_q * n_frames)
    return AudioPacket(seq=seq, codes=flat.reshape(n_frames, n_q).T.astype(np.int64))


def audio_payload_size(n_q: int, n_frames: int) -> int:
    return AUDIO_HEADER_SIZE + (n_q * n_frames * CODEBOOK_BITS + 7) // 8


def _bitpack10(values: np.ndarray) -> bytes:
    """Pack uint values < 1024 into a dense 10-bit big-endian bitstream."""
    v = np.asarray(values, dtype=np.uint16)
    # View as two bytes, swap to big-endian order, expand to 16 bits, keep low 10.
    bits = np.unpackbits(v.view(np.uint8).reshape(-1, 2)[:, ::-1], axis=1)
    flat = bits[:, 16 - CODEBOOK_BITS:].ravel()
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat).tobytes()


def _bitunpack10(blob: bytes, count: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
    need = count * CODEBOOK_BITS
    if bits.size < need:
        raise ValueError(f"audio payload short: {bits.size} bits < {need}")
    bits = bits[:need].reshape(count, CODEBOOK_BITS)
    pad = np.zeros((count, 16 - CODEBOOK_BITS), dtype=np.uint8)
    full = np.concatenate([pad, bits], axis=1)
    packed = np.packbits(full, axis=1)[:, ::-1].copy()   # back to little-endian
    return packed.view(np.uint16).ravel()


# ── Blob (reliable chunked transfer, used for the avatar JPEG) ────────────────

_BLOB_HEADER = struct.Struct("<BBBHH")      # magic, version, blob_id, chunk, total
BLOB_HEADER_SIZE = _BLOB_HEADER.size        # 7
BLOB_CHUNK_BYTES = 8 * 1024


def pack_blob(blob_id: int, payload: bytes, chunk_bytes: int = BLOB_CHUNK_BYTES):
    """Yield the chunk frames for one blob transfer."""
    total = max(1, (len(payload) + chunk_bytes - 1) // chunk_bytes)
    for i in range(total):
        piece = payload[i * chunk_bytes:(i + 1) * chunk_bytes]
        yield _BLOB_HEADER.pack(MAGIC_BLOB, VERSION, blob_id & 0xFF, i, total) + piece


class BlobReassembler:
    """Collects blob chunks; `feed` returns the payload once the last one lands."""

    def __init__(self):
        self._id: Optional[int] = None
        self._parts: dict[int, bytes] = {}
        self._total = 0

    def feed(self, data: bytes) -> Optional[bytes]:
        magic, version, blob_id, chunk, total = _BLOB_HEADER.unpack_from(data, 0)
        if magic != MAGIC_BLOB or version != VERSION:
            raise ValueError("not a blob chunk")
        if blob_id != self._id:
            self._id, self._parts, self._total = blob_id, {}, total
        self._parts[chunk] = data[BLOB_HEADER_SIZE:]
        if len(self._parts) < self._total:
            return None
        out = b"".join(self._parts[i] for i in range(self._total))
        self._id, self._parts, self._total = None, {}, 0
        return out


def peek_magic(data: bytes) -> int:
    return data[0] if data else 0
