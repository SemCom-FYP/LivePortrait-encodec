# coding: utf-8
"""
EnCodec as a conferencing audio codec.

EnCodec's 24 kHz model is not a streaming codec — it has no built-in framing —
so we drive it in fixed chunks and hand the raw residual-VQ codes to the wire
layer. At bandwidth 6.0 the quantiser keeps 8 codebooks at a 75 Hz frame rate,
which is 8 * 75 * 10 = 6 kbit/s of actual payload.

Encoding each chunk independently leaves a discontinuity at the seams, so the
encoder carries a short lookback of previous audio into the model and the
decoder drops it again. That gives the convolutional stack real context at the
chunk boundary instead of zero-padding.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio.functional as AF

from encodec import EncodecModel

SAMPLE_RATE = 24000
FRAME_RATE = 75.0               # EnCodec 24 kHz latent frames per second
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)   # 320
VALID_BANDWIDTHS = (1.5, 3.0, 6.0, 12.0, 24.0)


@dataclass
class CodecStats:
    kbps: float = 0.0
    n_q: int = 0
    bytes_last: int = 0
    encode_ms: float = 0.0
    decode_ms: float = 0.0


class EncodecStreamCodec:
    """Chunked EnCodec encode/decode. Encoder and decoder keep separate state."""

    def __init__(self, bandwidth: float = 6.0, device: str = "cpu",
                 chunk_ms: int = 240, lookback_ms: int = 80):
        if bandwidth not in VALID_BANDWIDTHS:
            raise ValueError(f"bandwidth must be one of {VALID_BANDWIDTHS}")

        self.device = torch.device(device)
        self.bandwidth = bandwidth
        self.model = EncodecModel.encodec_model_24khz().to(self.device).eval()
        self.model.set_target_bandwidth(bandwidth)

        self.n_q = self.model.quantizer.get_num_quantizers_for_bandwidth(
            self.model.frame_rate, bandwidth)
        self.nominal_kbps = self.n_q * self.model.frame_rate * 10 / 1000

        # Round chunk and lookback to whole latent frames so codes line up.
        self.chunk_samples = _round_frames(chunk_ms * SAMPLE_RATE // 1000)
        self.lookback_samples = _round_frames(lookback_ms * SAMPLE_RATE // 1000)
        self.lookback_frames = self.lookback_samples // SAMPLES_PER_FRAME

        self._enc_tail = np.zeros(self.lookback_samples, dtype=np.float32)
        self._resamplers: dict[int, None] = {}
        self.stats = CodecStats(n_q=self.n_q)

    # ── encoding ─────────────────────────────────────────────────────────────

    def encode(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """float32 mono PCM -> EnCodec codes (n_q, n_frames) covering only `pcm`."""
        t0 = time.perf_counter()
        pcm = self._to_24k(pcm, sample_rate)

        ctx = np.concatenate([self._enc_tail, pcm])
        self._enc_tail = pcm[-self.lookback_samples:] if self.lookback_samples else pcm[:0]
        if self._enc_tail.size < self.lookback_samples:
            self._enc_tail = np.pad(
                self._enc_tail, (self.lookback_samples - self._enc_tail.size, 0))

        wav = torch.from_numpy(ctx).to(self.device)[None, None, :]
        with torch.no_grad():
            frames = self.model.encode(wav)
        codes = frames[0][0][0]                      # (n_q, n_frames_with_lookback)
        codes = codes[:, self.lookback_frames:]      # drop the lookback region

        codes_np = codes.cpu().numpy()
        n_frames = codes_np.shape[1]
        duration = max(pcm.size / SAMPLE_RATE, 1e-6)
        self.stats.kbps = self.n_q * n_frames * 10 / duration / 1000
        self.stats.n_q = self.n_q
        self.stats.encode_ms = (time.perf_counter() - t0) * 1000
        return codes_np

    def _to_24k(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if sample_rate == SAMPLE_RATE:
            return pcm
        t = torch.from_numpy(pcm)[None, :]
        return AF.resample(t, sample_rate, SAMPLE_RATE).squeeze(0).numpy()

    # ── decoding ─────────────────────────────────────────────────────────────

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """EnCodec codes (n_q, n_frames) -> float32 mono PCM at 24 kHz."""
        t0 = time.perf_counter()
        c = torch.from_numpy(np.ascontiguousarray(codes)).long().to(self.device)[None]
        with torch.no_grad():
            wav = self.model.decode([(c, None)])
        out = wav[0, 0].float().cpu().numpy()
        self.stats.decode_ms = (time.perf_counter() - t0) * 1000
        return out

    def close(self) -> None:
        pass


def _round_frames(n_samples: int) -> int:
    return max(SAMPLES_PER_FRAME, (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME)


class CrossfadeJoiner:
    """Overlap-adds successive decoded chunks to hide seam discontinuities."""

    def __init__(self, fade_ms: int = 5):
        self.n = max(1, fade_ms * SAMPLE_RATE // 1000)
        self._ramp = np.linspace(0.0, 1.0, self.n, dtype=np.float32)
        self._tail: Optional[np.ndarray] = None

    def push(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size <= self.n:
            return pcm
        head, body, tail = pcm[:self.n], pcm[self.n:-self.n], pcm[-self.n:]
        if self._tail is None:
            out = np.concatenate([head, body])
        else:
            out = np.concatenate([self._tail * (1 - self._ramp) + head * self._ramp, body])
        self._tail = tail
        return out

    def reset(self) -> None:
        self._tail = None
