# coding: utf-8
"""
Speak into your mic, hear yourself translated in your OWN cloned voice —
one process, one machine, no camera, no signaling server, no second client.

    python -m webrtc_conference.chatterbox_mic_test --translate es

Same idea as mic_translate_test.py (see that file's docstring for why a
single-process mic-to-speaker loop is enough to test translation without the
full two-client WebRTC setup), but the voice is cloned from a short reference
clip of you instead of played back in one of Piper's fixed voices.

The reference clip is recorded from your own mic on first run (default 8 s —
just talk normally when it says "recording reference..."), or pass
--reference path/to/clip.wav to clone a specific voice instead (e.g. a
remote peer's voice saved from a real call). Cloning that clip into
Chatterbox's internal conditioning is a one-time cost (~1-3 s); after that
every sentence reuses it.

First run also downloads the Chatterbox Multilingual V3 checkpoint
(~3.1 GB) and takes several minutes just to finish writing/caching it —
that one-time cost is unrelated to anything below and only happens once.
Every run after is ~10 s to load.

Chatterbox trades speed for voice quality: synthesis runs at roughly
real-time (RTF ~0.8-1.0 on an RTX 4060), i.e. a 6 s sentence costs ~5 s of
GPU time to generate, versus Piper's ~0.1-0.3 s regardless of length. Total
lag per sentence is therefore noticeably longer than with Piper — watch the
printed "lag" numbers to judge whether that trade-off is acceptable for your
use. See webrtc_conference/translator.py's module docstring for the rest of
the pipeline (VAD -> streaming Whisper -> Opus-MT -> here).
"""

import argparse
import logging
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

log = logging.getLogger("chatterbox_mic_test")

CHATTERBOX_SR = 24000           # Chatterbox's own native rate; happens to match ours


class ChatterboxVoice:
    """Wraps Chatterbox Multilingual V3 as a `speak_fn` for PeerTranslation.

    The model is shared (one GPU instance), but `model.conds` — which voice
    it's currently cloned to — is a single mutable attribute. `_lock` makes
    "set the voice, then generate" atomic, so this is safe even if it were
    ever used for more than one peer/voice at a time.
    """

    def __init__(self, device: str = "auto", max_new_tokens: int = 700):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        dev = device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading Chatterbox Multilingual V3 on %s …", dev)
        t0 = time.time()
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=dev, t3_model="v3")
        log.info("Chatterbox ready in %.1fs (native rate %d Hz)", time.time() - t0, self.model.sr)
        self._lock = threading.Lock()
        self._conds = None
        # generate()'s own call hardcodes max_new_tokens=1000 with no way to
        # pass a smaller one through; on text that confuses it into never
        # emitting EOS (seen on "A.S.R.") that runs the full 1000 steps,
        # ~45-50s on an RTX 4060 for what should be well under a second of
        # audio. Calling t3.inference()/s3gen.inference() directly, exactly
        # as generate() does internally, lets this cap stay well short of
        # that — real sentences in this pipeline finish in under 300 steps
        # (see translator.py's StreamingTranscriber, which already keeps
        # sentences short), so this only ever bites the pathological case.
        self.max_new_tokens = max_new_tokens

    def _generate_bounded(self, text: str, language_id: str):
        """Same steps as ChatterboxMultilingualTTS.generate(), down to the
        trailing-token trim and watermark, but with a bounded max_new_tokens
        — see the comment in __init__ for why generate() itself isn't used."""
        import torch.nn.functional as F
        from chatterbox.mtl_tts import punc_norm
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.models.s3tokenizer import S3_TOKEN_RATE, drop_invalid_tokens

        m = self.model
        text = punc_norm(text)
        text_tokens = m.tokenizer.text_to_tokens(text, language_id=language_id.lower()).to(m.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)   # two seqs for CFG
        text_tokens = F.pad(text_tokens, (1, 0), value=m.t3.hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=m.t3.hp.stop_text_token)

        with torch.inference_mode():
            speech_tokens = m.t3.inference(
                t3_cond=m.conds.t3, text_tokens=text_tokens,
                max_new_tokens=self.max_new_tokens, temperature=0.8,
                cfg_weight=0.5, repetition_penalty=1.2, min_p=0.05, top_p=1.0)
            speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(m.device)
            wav, _ = m.s3gen.inference(speech_tokens=speech_tokens, ref_dict=m.conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()
            # Drop the final token's audio: emitted just before EOS with
            # degraded attention, decodes to ~40ms of noise (generate()
            # does the same trim).
            st_len = max(1, int(speech_tokens.shape[-1]) - 1)
            wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]
            watermarked = m.watermarker.apply_watermark(wav, sample_rate=m.sr)
        return torch.from_numpy(watermarked).unsqueeze(0)

    def clone(self, wav_path: str):
        """One-time per voice: extract conditioning from a reference clip."""
        t0 = time.time()
        with self._lock:
            self.model.prepare_conditionals(wav_path)
            self._conds = self.model.conds
        log.info("cloned voice from %s in %.2fs", wav_path, time.time() - t0)

    def speak(self, text: str, language_id: str, speed: float = 1.0):
        """Matches PeerTranslation's speak_fn(text, code, speed) contract."""
        if self._conds is None:
            raise RuntimeError("clone() a reference voice before speak()")
        with self._lock:
            self.model.conds = self._conds
            wav = self._generate_bounded(text, language_id)
        pcm = wav.squeeze().float().cpu().numpy()
        if speed != 1.0:
            # Chatterbox has no built-in rate control; resampling the result
            # is a cheap way to speed up playback (pitch shifts slightly,
            # same trade-off python's audioop/ffmpeg atempo-free tricks make).
            n = max(1, int(pcm.size / speed))
            pcm = np.interp(np.linspace(0, pcm.size - 1, n), np.arange(pcm.size), pcm).astype(np.float32)
        if self.model.sr != SAMPLE_RATE:
            pcm = AF.resample(torch.from_numpy(pcm)[None], self.model.sr, SAMPLE_RATE)[0].numpy()
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

    path = Path(tempfile.gettempdir()) / "chatterbox_mic_test_reference.wav"
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
    p.add_argument("--chatterbox-device", choices=["auto", "cuda", "cpu"], default="auto")
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

    voice = ChatterboxVoice(device=args.chatterbox_device)
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
