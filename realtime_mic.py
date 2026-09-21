#!/usr/bin/env python3
"""Real-time microphone demo: EnCodec vs Opus side-by-side bitrate comparison."""

import sys
import time
import queue
import ctypes
import threading
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import sounddevice as sd
import torchaudio.functional as F

from encodec import EncodecModel
from encodec.utils import convert_audio
import pyogg.opus as _op


# ──────────────────────────────────────────────────────────────────────────────
#  Codec abstractions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CodecStats:
    actual_bw: float = 0.0    # kbps
    n_q: int = 0              # codebooks (EnCodec) or frames (Opus)
    compressed_bytes: int = 0
    latency_ms: float = 0.0


class Codec(ABC):
    name: str
    target_bw: float          # kbps

    @abstractmethod
    def encode_decode(self, wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, CodecStats]:
        """wav: float32 mono [T].  Returns (decoded_float32 [T], stats)."""
        ...

    @abstractmethod
    def close(self) -> None: ...


class EncodecCodec(Codec):
    name = "EnCodec"

    def __init__(self, bandwidth: float):
        self.target_bw = bandwidth
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        self._sr = self.model.sample_rate          # 24 000
        self._ch = self.model.channels             # 1
        # n_q for this bandwidth
        self._n_q = self.model.quantizer.get_num_quantizers_for_bandwidth(
            self.model.frame_rate, bandwidth)
        BITS = 10  # log2(1024 codebook bins)
        self._theoretical_bw = self._n_q * self.model.frame_rate * BITS / 1000

    @property
    def sample_rate(self) -> int:
        return self._sr

    def encode_decode(self, wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, CodecStats]:
        t0 = time.perf_counter()
        tensor = torch.from_numpy(wav).float().unsqueeze(0)   # [1, T]
        tensor = convert_audio(tensor, sample_rate, self._sr, self._ch)
        batch  = tensor.unsqueeze(0)                           # [1, 1, T]

        with torch.no_grad():
            frames  = self.model.encode(batch)
            decoded = self.model.decode(frames)

        codes, _ = frames[0]
        n_q     = codes.shape[1]
        T_fr    = codes.shape[2]
        dur     = tensor.shape[-1] / self._sr
        bw      = n_q * T_fr * 10 / dur / 1000
        comp    = n_q * T_fr * 10 // 8  # bytes (10 bits per entry)

        out = decoded[0, 0].numpy()
        return out, CodecStats(
            actual_bw=bw,
            n_q=n_q,
            compressed_bytes=comp,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def close(self) -> None:
        pass


class OpusCodec(Codec):
    name = "Opus"
    # Opus supports: 8000, 12000, 16000, 24000, 48000 Hz
    OPUS_SR = 24000
    FRAME_MS = 20
    MAX_PACKET = 4000

    def __init__(self, bitrate_kbps: float):
        self.target_bw = bitrate_kbps
        bitrate_bps = int(bitrate_kbps * 1000)
        self.FRAME_SIZE = self.OPUS_SR * self.FRAME_MS // 1000   # 480 samples

        err = ctypes.c_int(0)
        self._enc = _op.opus_encoder_create(
            self.OPUS_SR, 1, _op.OPUS_APPLICATION_AUDIO, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"Opus encoder creation failed: {err.value}")
        _op.opus_encoder_ctl(self._enc, _op.OPUS_SET_BITRATE_REQUEST,
                             ctypes.c_int(bitrate_bps))

        self._dec = _op.opus_decoder_create(
            self.OPUS_SR, 1, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"Opus decoder creation failed: {err.value}")

        self._out_buf = (ctypes.c_ubyte * self.MAX_PACKET)()

    @property
    def sample_rate(self) -> int:
        return self.OPUS_SR

    def encode_decode(self, wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, CodecStats]:
        t0 = time.perf_counter()

        # Resample to Opus SR if needed
        if sample_rate != self.OPUS_SR:
            tensor = torch.from_numpy(wav).float().unsqueeze(0)
            tensor = F.resample(tensor, sample_rate, self.OPUS_SR)
            wav = tensor.squeeze(0).numpy()

        # Pad to a multiple of FRAME_SIZE
        pad = (-len(wav)) % self.FRAME_SIZE
        if pad:
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])

        pcm_int16  = (wav * 32767).astype(np.int16)
        total_bytes = 0
        decoded_chunks = []

        for i in range(0, len(pcm_int16), self.FRAME_SIZE):
            frame = pcm_int16[i : i + self.FRAME_SIZE]
            pcm_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

            n = _op.opus_encode(self._enc, pcm_ptr, self.FRAME_SIZE,
                                self._out_buf, self.MAX_PACKET)
            if n < 0:
                n = 0

            total_bytes += n

            out_pcm = np.zeros(self.FRAME_SIZE, dtype=np.int16)
            out_ptr = out_pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            _op.opus_decode(self._dec, self._out_buf, n,
                            out_ptr, self.FRAME_SIZE, 0)
            decoded_chunks.append(out_pcm.astype(np.float32) / 32767.0)

        decoded = np.concatenate(decoded_chunks)[:len(wav) - pad if pad else len(wav)]
        dur = len(wav) / self.OPUS_SR
        bw  = total_bytes * 8 / dur / 1000  # kbps

        return decoded, CodecStats(
            actual_bw=bw,
            n_q=len(decoded_chunks),          # re-purpose n_q as frame count
            compressed_bytes=total_bytes,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def close(self) -> None:
        if self._enc:
            _op.opus_encoder_destroy(self._enc)
        if self._dec:
            _op.opus_decoder_destroy(self._dec)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def vu_bar(rms: float, width: int = 20) -> str:
    db     = 20 * np.log10(rms + 1e-9)
    filled = int(np.clip((db + 60) / 60 * width, 0, width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time EnCodec + Opus microphone demo")
    p.add_argument("--codec", choices=["encodec", "opus", "both"], default="both",
                   help="Which codec(s) to run (default: both)")
    p.add_argument("--bandwidth", type=float, default=6.0,
                   choices=[1.5, 3.0, 6.0, 12.0, 24.0],
                   help="EnCodec target bandwidth in kbps (default: 6.0)")
    p.add_argument("--opus-bitrate", type=float, default=None,
                   help="Opus target bitrate in kbps (default: same as --bandwidth)")
    p.add_argument("--chunk", type=float, default=0.5,
                   help="Chunk duration in seconds (default: 0.5)")
    p.add_argument("--playback", choices=["encodec", "opus", "none"], default="none",
                   help="Which codec's output to play back (default: none)")
    p.add_argument("--device", type=int, default=None,
                   help="Input device index")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio devices and exit")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    opus_kbps = args.opus_bitrate if args.opus_bitrate is not None else args.bandwidth

    # Build codec list
    codecs: list[Codec] = []
    print("Initialising codecs …")
    if args.codec in ("encodec", "both"):
        print(f"  Loading EnCodec 24 kHz  (bandwidth={args.bandwidth} kbps) …")
        codecs.append(EncodecCodec(args.bandwidth))
    if args.codec in ("opus", "both"):
        print(f"  Creating Opus encoder   (bitrate={opus_kbps} kbps) …")
        codecs.append(OpusCodec(opus_kbps))

    device_info  = sd.query_devices(args.device, "input")
    MIC_SR       = int(device_info["default_samplerate"])
    CHUNK_SAMPLES = int(args.chunk * MIC_SR)
    RAW_KBPS     = MIC_SR * 32 / 1000   # 32-bit float mono

    audio_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
    playback_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3)

    # Per-codec live stats (indexed same as `codecs`)
    live_stats: list[CodecStats] = [CodecStats() for _ in codecs]
    live_rms = [0.0]

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"\nMic: {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())   # mono float32

    def process_loop():
        while True:
            wav = audio_q.get()
            if wav is None:
                break

            live_rms[0] = float(np.sqrt(np.mean(wav ** 2)))
            last_decoded: Optional[np.ndarray] = None
            last_sr = MIC_SR

            for idx, codec in enumerate(codecs):
                decoded, stats = codec.encode_decode(wav, MIC_SR)
                live_stats[idx] = stats

                if args.playback == codec.name.lower():
                    last_decoded = decoded
                    last_sr      = codec.sample_rate

            if last_decoded is not None and not playback_q.full():
                playback_q.put((last_decoded.astype(np.float32), last_sr))

    def display_loop():
        NAMES = [c.name for c in codecs]
        TARGETS = [c.target_bw for c in codecs]
        while True:
            bar = vu_bar(live_rms[0])
            lines = [f"\r  {bar}  RMS {live_rms[0]:.4f}"]
            for i, (name, tgt) in enumerate(zip(NAMES, TARGETS)):
                s = live_stats[i]
                ratio = RAW_KBPS / s.actual_bw if s.actual_bw > 0 else 0
                unit  = "codebooks" if name == "EnCodec" else "frames"
                lines.append(
                    f"    {name:<8} BW: {s.actual_bw:6.2f} kbps "
                    f"(tgt {tgt:.1f})  "
                    f"{unit}: {s.n_q:3d}  "
                    f"bytes: {s.compressed_bytes:5d}  "
                    f"ratio: {ratio:4.0f}x  "
                    f"lat: {s.latency_ms:4.0f} ms"
                )
            print("\n".join(lines) + "   ", end="", flush=True)
            # Move cursor back up so next update overwrites same lines
            n_extra = len(lines) - 1
            if n_extra:
                print(f"\033[{n_extra}A", end="", flush=True)
            time.sleep(0.1)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n  Mic device : {device_info['name']}  ({MIC_SR} Hz)")
    print(f"  Chunk      : {args.chunk}s  ({CHUNK_SAMPLES} samples)")
    print(f"  Raw PCM    : {RAW_KBPS:.0f} kbps  (mono 32-bit float)")
    if isinstance(codecs[0] if codecs else None, EncodecCodec):
        ec = next((c for c in codecs if isinstance(c, EncodecCodec)), None)
        if ec:
            print(f"  EnCodec    : {ec._n_q} codebooks → ~{ec._theoretical_bw:.2f} kbps")
    print(f"\n  Press Ctrl+C to stop.\n")

    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=display_loop, daemon=True).start()

    out_stream: Optional[sd.OutputStream] = None
    if args.playback != "none":
        # Output SR determined at runtime from first decoded frame
        out_stream = sd.OutputStream(samplerate=24000, channels=1)
        out_stream.start()

    try:
        with sd.InputStream(
            samplerate=MIC_SR, channels=1, blocksize=CHUNK_SAMPLES,
            device=args.device, callback=mic_callback,
        ):
            while True:
                if out_stream is not None:
                    try:
                        pcm, _ = playback_q.get(timeout=0.5)
                        out_stream.write(pcm.reshape(-1, 1))
                    except queue.Empty:
                        pass
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n\nStopped.")
    finally:
        audio_q.put(None)
        if out_stream:
            out_stream.stop()
            out_stream.close()
        for c in codecs:
            c.close()


if __name__ == "__main__":
    main()
