# coding: utf-8
"""
Speak into your mic, hear yourself translated in your OWN cloned voice —
same one-process mic-to-speaker loop as pocket_tts_mic_test.py (see that
file's docstring for why this is enough to test the pipeline without the
full WebRTC setup), but recognition is done by Alibaba's Qwen3-ASR-0.6B
instead of Whisper, alongside Kyutai's Pocket TTS for the cloned voice:

    python -m webrtc_conference.qwen3_asr_pocket_tts_mic_test --translate es

Why this isn't just "pass asr_model=Qwen3-ASR to translator.py"
-----------------------------------------------------------------
translator.py's StreamingTranscriber is built around faster-whisper
specifically: it re-reads a growing audio buffer every second or so and
only accepts a word once two readings agree on it (LocalAgreement-2), using
Whisper's word-level timestamps to align readings and its no_speech_prob /
avg_logprob / compression_ratio to drop hallucinations. Qwen3-ASR-0.6B
exposes none of that — no word timestamps, no confidence scores, and it
isn't meant to be re-run incrementally on a half-finished utterance (word
timestamps need a *second* model, Qwen3-ForcedAligner, which this script
doesn't use since nothing here needs them).

So here recognition is reduced to what both models genuinely support: pure
boundary detection, same Silero VAD and (optionally) pipecat-ai's smart-turn
used elsewhere in this project to decide where an utterance starts and
ends, and only once it ends is the *whole* utterance handed to Qwen3-ASR in
a single call — see Qwen3Transcriber below. That costs one Qwen3-ASR call
per sentence rather than one per second of speech in progress (which matters
since Qwen3-ASR is slower per call than faster-whisper's `small` on the same
hardware), at the cost of losing StreamingTranscriber's word-by-word partial
confirmation and mid-sentence clause splitting: nothing is transcribed,
translated, or spoken until the speaker actually pauses.

Everything else — Opus-MT translation, Pocket TTS voice cloning, smart-turn
endpointing, the pacing that speeds up the translated voice to avoid falling
behind — is unchanged from pocket_tts_mic_test.py; this script only swaps
out what sits between "the speaker paused" and "here is what they said".

Needs the `qwen-asr` package (not in requirements.txt, optional):
    pip install -U qwen-asr
First run downloads the ~0.6B-parameter checkpoint (small) from the Hugging
Face Hub. GPU is used automatically when available (bfloat16); CPU works
too, just slower per utterance — watch the printed "lag" numbers.
"""

import argparse
import logging
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torchaudio.functional as AF

from ..audio_codec import SAMPLE_RATE
from ..media import MicReader, SpeakerMixer, list_devices, probe_input_rate
from .pocket_tts_mic_test import PocketTTSVoice, record_reference
from ..translator import (ASR_RATE, HALLUCINATION_PHRASES, LANGUAGES, VOICE_BUFFER_S, _STRIP,
                         PeerTranslation, Sentence, SpeechTranslator, StreamingVAD,
                         TurnDetector)

log = logging.getLogger("qwen3_asr_mic_test")


class Qwen3ASR:
    """Wraps Alibaba's Qwen3-ASR-0.6B (via the `qwen-asr` package) as a
    whole-utterance transcriber: one call in, one transcript out — no
    streaming, no word timestamps, no confidence score. See the module
    docstring for why this project's usual incremental Whisper recognition
    doesn't apply here."""

    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-0.6B", device: str = "auto",
                 dtype: str = "auto", language: Optional[str] = "English"):
        from qwen_asr import Qwen3ASRModel
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        else:
            dtype = getattr(torch, dtype)
        self.language = None if language in (None, "", "auto") else language
        log.info("loading Qwen3-ASR %s on %s (%s) …", model_id, device, dtype)
        t0 = time.time()
        self.model = Qwen3ASRModel.from_pretrained(model_id, dtype=dtype, device_map=device)
        log.info("Qwen3-ASR ready in %.1fs", time.time() - t0)

    def transcribe(self, audio16k: np.ndarray) -> str:
        results = self.model.transcribe(audio=(audio16k.astype(np.float32, copy=False), ASR_RATE),
                                        language=self.language)
        return results[0].text.strip() if results else ""


class Qwen3Transcriber:
    """Silero VAD (+ optional smart-turn) marks where an utterance starts and
    ends; only the finished utterance is handed to Qwen3-ASR, once. Offers
    the same `push(pcm16k) -> list[Sentence]` / `reset()` contract as
    translator.py's StreamingTranscriber so it drops straight into
    PeerTranslation's `transcriber` parameter — see that class and the
    module docstring above for why the two aren't interchangeable in their
    internals, only at this boundary.
    """

    START_TH, END_TH = 0.5, 0.35        # VAD speech probability
    PAD = 3                             # windows of silence kept around speech

    def __init__(self, asr: Qwen3ASR, end_pause_s: float = 0.8,
                 max_utterance_s: float = 20.0,
                 turn_detector: Optional[TurnDetector] = None,
                 turn_check_s: float = 0.3, turn_complete_th: float = 0.5):
        self.asr = asr
        self.vad = StreamingVAD()
        self.win_s = StreamingVAD.WINDOW / ASR_RATE
        self.end_pause = round(end_pause_s / self.win_s)
        self.max_wins = round(max_utterance_s / self.win_s)
        self.turn_detector = turn_detector
        self.turn_check = round(turn_check_s / self.win_s)
        self.turn_complete_th = turn_complete_th
        self.reset()

    def reset(self):
        self.vad.reset()
        self._t = 0.0
        self._arrivals: deque = deque(maxlen=512)       # (stream t, perf_counter)
        self._preroll: deque = deque(maxlen=self.PAD)
        self._wins: list = []
        self._buf_t0 = 0.0
        self._active = False
        self._silence = 0
        self._turn_checked = False

    def push(self, pcm16k: np.ndarray) -> list:
        self._arrivals.append((self._t + pcm16k.size / ASR_RATE, time.perf_counter()))
        out = []
        for win, p in self.vad(pcm16k):
            self._t += self.win_s
            if not self._active:
                self._preroll.append(win)
                if p >= self.START_TH:
                    self._active = True
                    self._wins = list(self._preroll)
                    self._buf_t0 = self._t - len(self._wins) * self.win_s
                    self._silence, self._turn_checked = 0, False
                continue
            self._wins.append(win)
            if p < self.END_TH:
                if self._silence == 0:
                    self._turn_checked = False
                self._silence += 1
            else:
                self._silence = 0
            finished = self._silence >= self.end_pause
            if not finished and self.turn_detector is not None and not self._turn_checked \
                    and self._silence >= self.turn_check:
                self._turn_checked = True
                finished = self._turn_sounds_finished()
            if not finished and len(self._wins) >= self.max_wins:
                # No pause at all after a long time: cut now rather than make
                # the speaker wait minutes for a translation that keeps
                # growing (and Qwen3-ASR was never asked to read this long).
                finished = True
            if finished:
                s = self._finish()
                if s is not None:
                    out.append(s)
        return out

    def _turn_sounds_finished(self) -> bool:
        try:
            prob = self.turn_detector.complete_probability(np.concatenate(self._wins))
        except Exception as exc:
            log.warning("smart-turn inference failed: %s", exc)
            return False
        return prob >= self.turn_complete_th

    def _finish(self) -> Optional[Sentence]:
        wins = self._wins
        if self._silence:
            # Drop the trailing silence beyond a small pad; Qwen3-ASR hears
            # the pause as noise otherwise, for no benefit.
            wins = wins[:max(0, len(wins) - self._silence + self.PAD)]
        start_t = self._buf_t0
        self._active, self._wins = False, []
        self._preroll.clear()
        if not wins:
            return None
        audio = np.concatenate(wins)
        end_t = start_t + audio.size / ASR_RATE
        try:
            text = self.asr.transcribe(audio)
        except Exception as exc:
            log.warning("Qwen3-ASR failed: %s", exc)
            return None
        # No confidence score to filter hallucinations with (see module
        # docstring), so only the same known silence/noise phrases
        # translator.py's Whisper path drops are checked here too.
        if not text or text.strip().lower().translate(_STRIP) in HALLUCINATION_PHRASES:
            return None
        heard = next((wall for t, wall in self._arrivals if t >= end_t), time.perf_counter())
        return Sentence(text, start_t, end_t, heard)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--translate", choices=sorted(LANGUAGES), default="es")
    p.add_argument("--reference", default=None,
                   help="Wav file, named Pocket TTS voice, or hf:// path to clone "
                        "(default: record 8s from your mic)")
    p.add_argument("--reference-seconds", type=float, default=8.0)
    p.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--qwen-device", choices=["auto", "cuda:0", "cpu"], default="auto")
    p.add_argument("--qwen-dtype", choices=["auto", "float32", "float16", "bfloat16"],
                   default="auto")
    p.add_argument("--qwen-language", default="English",
                   help="forcing the source language is faster and more accurate than "
                        "Qwen3-ASR's auto-detection; pass 'auto' to detect instead")
    p.add_argument("--mt-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="device for Opus-MT translation (independent of --qwen-device)")
    p.add_argument("--pocket-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="auto stays on CPU; GPU support is unofficial (see Pocket TTS docs)")
    p.add_argument("--pocket-quantize", action="store_true",
                   help="int8 dynamic quantization, CPU only: smaller/faster, same quality")
    p.add_argument("--mt-model", choices=["base", "big"], default="base")
    p.add_argument("--translate-threads", type=int, default=4)
    p.add_argument("--no-turn-detection", action="store_true",
                   help="disable smart-turn (https://github.com/pipecat-ai/smart-turn) "
                        "semantic turn detection; fall back to a fixed silence pause alone")
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--audio-chunk-ms", type=int, default=100)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s  %(name)-10s %(message)s", datefmt="%H:%M:%S")

    if args.list_devices:
        print(list_devices())
        return 0

    lang = LANGUAGES[args.translate]
    in_rate = probe_input_rate(args.input_device)

    reference = args.reference or record_reference(args.reference_seconds,
                                                    args.input_device, in_rate)

    print("Loading Qwen3-ASR …")
    qwen = Qwen3ASR(args.qwen_model, device=args.qwen_device, dtype=args.qwen_dtype,
                    language=args.qwen_language)

    print("Loading Opus-MT …")
    translator = SpeechTranslator(asr_model=None, device=args.mt_device,
                                  threads=args.translate_threads, mt_size=args.mt_model,
                                  load_tts=False)
    translator.prepare(args.translate)

    turn_detector = None
    if not args.no_turn_detection:
        try:
            print("Loading smart-turn (turn detection) …")
            turn_detector = TurnDetector()
        except Exception as exc:
            log.warning("smart-turn unavailable (%s); falling back to a fixed "
                        "silence pause for sentence endpointing", exc)

    voice = PocketTTSVoice(device=args.pocket_device, quantize=args.pocket_quantize)
    voice.clone(reference)

    print(f"Ready. Speaking English -> {lang.name}, in the cloned voice. Ctrl+C to stop.\n")

    speaker = SpeakerMixer(SAMPLE_RATE, device=args.output_device)
    speaker.start()
    key = "mic"

    def on_pcm(pcm: np.ndarray):
        speaker.push(key, pcm, cap_s=VOICE_BUFFER_S)

    transcriber = Qwen3Transcriber(qwen, turn_detector=turn_detector)
    translation = PeerTranslation(translator, "you", args.translate,
                                  out=on_pcm, backlog_ms=lambda: speaker.backlog_ms(key),
                                  speak_fn=voice.speak, transcriber=transcriber)

    def on_mic_chunk(pcm: np.ndarray):
        resampled = pcm if in_rate == SAMPLE_RATE else AF.resample(
            torch.from_numpy(pcm)[None], in_rate, SAMPLE_RATE)[0].numpy()
        translation.feed(resampled)

    chunk_samples = int(in_rate * args.audio_chunk_ms / 1000)
    mic = MicReader(on_mic_chunk, chunk_samples=chunk_samples,
                    device=args.input_device, sample_rate=in_rate)
    mic.start()
    print(f"mic: {mic.name} @ {mic.sample_rate} Hz")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping …")
        translation.close()
        mic.close()
        speaker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
