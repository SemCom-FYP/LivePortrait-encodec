# coding: utf-8
"""
Speak into your mic, hear yourself translated in your OWN cloned voice —
one process, one machine, no camera, no signaling server, no second client.

    python -m webrtc_conference.xttsv2_mic_test --translate es

Same idea as chatterbox_mic_test.py (see that file's docstring, and
mic_translate_test.py's, for why a single-process mic-to-speaker loop is
enough to test translation without the full two-client WebRTC setup), but
the cloned voice is produced by Coqui XTTS v2 instead of Chatterbox.

The reference clip is recorded from your own mic on first run (default 8 s —
just talk normally when it says "recording reference..."), or pass
--reference path/to/clip.wav to clone a specific voice instead (e.g. a
remote peer's voice saved from a real call). Cloning that clip into XTTS's
conditioning latents is a one-time cost (~1-2 s); after that every sentence
reuses it.

First run also downloads the XTTS v2 checkpoint (~1.8 GB). Every run after
is a few seconds to load.

Needs the `TTS` package, which is NOT in requirements.txt (optional, heavy):
    pip install coqui-tts
(the original `TTS` PyPI package is unmaintained and caps out around Python
3.11; the `coqui-tts` fork is a drop-in replacement — same `TTS` import —
that keeps up with current Python/torch.) Downloading the model agrees to
Coqui's non-commercial license on your behalf non-interactively (see
XTTSVoice.__init__); read it before use if that matters to you:
https://coqui.ai/cpml

XTTS trades speed for voice quality similarly to Chatterbox: synthesis is
roughly real-time on a modern GPU, noticeably slower on CPU. Total lag per
sentence is therefore longer than with Piper — watch the printed "lag"
numbers to judge whether that trade-off is acceptable for your use. See
webrtc_conference/translator.py's module docstring for the rest of the
pipeline (VAD -> streaming Whisper -> Opus-MT -> here).
"""

import argparse
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from ..audio_codec import SAMPLE_RATE
from ..media import MicReader, SpeakerMixer, list_devices, probe_input_rate
from ..translator import LANGUAGES, VOICE_BUFFER_S, PeerTranslation, SpeechTranslator

log = logging.getLogger("xttsv2_mic_test")

XTTS_SR = 24000           # XTTS v2's own native rate; happens to match ours


class XTTSVoice:
    """Wraps Coqui XTTS v2 as a `speak_fn` for PeerTranslation.

    The model is shared (one GPU instance), but its conditioning latents —
    which voice it's currently cloned to — are a single mutable attribute.
    `_lock` makes "set the voice, then generate" atomic, so this is safe
    even if it were ever used for more than one peer/voice at a time.
    """

    def __init__(self, device: str = "auto"):
        # Downloading the checkpoint otherwise blocks on an interactive y/N
        # prompt agreeing to Coqui's non-commercial license (CPML); running
        # this script at all is treated as agreeing to it — see the module
        # docstring for the license link.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS
        dev = device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading XTTS v2 on %s …", dev)
        t0 = time.time()
        self.api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
        self.model = self.api.synthesizer.tts_model
        self.sr = self.api.synthesizer.output_sample_rate
        log.info("XTTS ready in %.1fs (native rate %d Hz)", time.time() - t0, self.sr)
        self._lock = threading.Lock()
        self._latents = None

    def clone(self, wav_path: str):
        """One-time per voice: extract conditioning latents from a reference clip."""
        t0 = time.time()
        with self._lock:
            self._latents = self.model.get_conditioning_latents(audio_path=[wav_path])
        log.info("cloned voice from %s in %.2fs", wav_path, time.time() - t0)

    def speak(self, text: str, language_id: str, speed: float = 1.0):
        """Matches PeerTranslation's speak_fn(text, code, speed) contract."""
        if self._latents is None:
            raise RuntimeError("clone() a reference voice before speak()")
        gpt_cond_latent, speaker_embedding = self._latents
        with self._lock:
            with torch.inference_mode():
                out = self.model.inference(
                    text, language_id.lower(), gpt_cond_latent, speaker_embedding,
                    temperature=0.7, repetition_penalty=5.0)
        pcm = np.asarray(out["wav"], dtype=np.float32)
        if speed != 1.0:
            # XTTS has no reliable built-in rate control across versions;
            # resampling the result is a cheap way to speed up playback
            # (pitch shifts slightly, same trade-off python's audioop/ffmpeg
            # atempo-free tricks make — see chatterbox_mic_test.py).
            n = max(1, int(pcm.size / speed))
            pcm = np.interp(np.linspace(0, pcm.size - 1, n), np.arange(pcm.size), pcm).astype(np.float32)
        if self.sr != SAMPLE_RATE:
            pcm = AF.resample(torch.from_numpy(pcm)[None], self.sr, SAMPLE_RATE)[0].numpy()
        yield pcm


def record_reference(seconds: float, device, sample_rate: int) -> str:
    """Records `seconds` of mic audio to a temp wav file and returns its path."""
    print(f"Recording a {seconds:.0f}s voice reference — talk normally now …")
    chunks = []
    done = threading.Event()

    def on_chunk(pcm: np.ndarray):
        chunks.append(pcm.copy())
        if sum(c.size for c in chunks) >= seconds * sample_rate:
            done.set()

    mic = MicReader(on_chunk, chunk_samples=int(sample_rate * 0.1),
                    device=device, sample_rate=sample_rate)
    mic.start()
    done.wait(timeout=seconds + 5)
    mic.close()
    pcm = np.concatenate(chunks) if chunks else np.zeros(sample_rate, np.float32)

    path = Path(tempfile.gettempdir()) / "xttsv2_mic_test_reference.wav"
    sf.write(str(path), pcm, sample_rate)
    print(f"Reference saved to {path} ({pcm.size / sample_rate:.1f}s)\n")
    return str(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--translate", choices=sorted(LANGUAGES), default="es")
    p.add_argument("--reference", default=None,
                   help="Wav file of the voice to clone (default: record 8s from your mic)")
    p.add_argument("--reference-seconds", type=float, default=8.0)
    p.add_argument("--asr-model", default="small")
    p.add_argument("--asr-device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--xtts-device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--mt-model", choices=["base", "big"], default="base")
    p.add_argument("--translate-threads", type=int, default=4)
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

    print("Loading Whisper + Opus-MT …")
    translator = SpeechTranslator(args.asr_model, device=args.asr_device,
                                  threads=args.translate_threads, mt_size=args.mt_model)
    log.info("speech recognition: whisper %s on %s", args.asr_model, translator.asr_device)
    translator.prepare(args.translate)

    voice = XTTSVoice(device=args.xtts_device)
    voice.clone(reference)

    print(f"Ready. Speaking English -> {lang.name}, in the cloned voice. Ctrl+C to stop.\n")

    speaker = SpeakerMixer(SAMPLE_RATE, device=args.output_device)
    speaker.start()
    key = "mic"

    def on_pcm(pcm: np.ndarray):
        speaker.push(key, pcm, cap_s=VOICE_BUFFER_S)

    translation = PeerTranslation(translator, "you", args.translate,
                                  out=on_pcm, backlog_ms=lambda: speaker.backlog_ms(key),
                                  speak_fn=voice.speak)

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
