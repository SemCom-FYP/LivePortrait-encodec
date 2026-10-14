# coding: utf-8
"""
Speak into your mic, hear yourself translated in your OWN cloned voice —
one process, one machine, no camera, no signaling server, no second client.

    python -m webrtc_conference.pocket_tts_mic_test --translate es

Same idea as chatterbox_mic_test.py / xttsv2_mic_test.py (see those files'
docstrings, and mic_translate_test.py's, for why a single-process
mic-to-speaker loop is enough to test translation without the full
two-client WebRTC setup), but the cloned voice is produced by Kyutai's
Pocket TTS instead of Chatterbox or XTTS.

Pocket TTS is the odd one out of the three: a 100M-parameter model designed
to run on a CPU (no GPU required, ~2-6x real-time on a laptop CPU), with a
separate checkpoint per language rather than one multilingual model — so
unlike ChatterboxVoice/XTTSVoice there isn't a single shared model instance
here; PocketTTSVoice loads one TTSModel per target language the first time
it's needed and clones the same reference clip into each. Every language in
translator.py's LANGUAGES (es, fr, de, it) has a matching Pocket TTS model,
so --translate isn't restricted the way openvoice_mic_test.py's is.

The reference clip is recorded from your own mic on first run (default 8 s
— just talk normally when it says "recording reference..."), or pass
--reference path/to/clip.wav to clone a specific voice instead. --reference
can also be one of Pocket TTS's own named voices (e.g. "alba") or an
hf://kyutai/tts-voices/... path — see https://huggingface.co/kyutai/tts-voices.
Cloning is a one-time cost per language (~1-2 s on CPU); after that every
sentence in that language reuses it.

First run downloads each language's checkpoint (small; ~100M params) the
first time that language is spoken. Needs the `pocket-tts` package, which
is NOT in requirements.txt (optional):
    pip install pocket-tts
On Linux this pulls PyTorch's CUDA wheels by default even though Pocket TTS
runs on CPU (adds a few GB for nothing); use
    pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu
to avoid that. GPU use is unofficial (see --pocket-device below).

Pocket TTS trades voice quality for speed in the other direction from
Chatterbox/XTTS: it's the lightest and fastest of the three (runs fine with
no GPU at all), at a smaller 100M-parameter model's quality ceiling. Watch
the printed "lag" numbers as with the other two scripts. See
webrtc_conference/translator.py's module docstring for the rest of the
pipeline (VAD -> streaming Whisper -> Opus-MT -> here).

By default this script also loads pipecat-ai's smart-turn
(https://github.com/pipecat-ai/smart-turn) to decide when a pause is really
the end of your turn, rather than only waiting out a fixed silence — see
translator.py's TurnDetector. It downloads a small (~9 MB) ONNX checkpoint
from the Hugging Face Hub on first use; pass --no-turn-detection to skip it
and fall back to the fixed pause alone.
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
from ..translator import (LANGUAGES, VOICE_BUFFER_S, PeerTranslation, PocketTTSVoice,
                          SpeechTranslator, TurnDetector)

log = logging.getLogger("pocket_tts_mic_test")


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

    path = Path(tempfile.gettempdir()) / "pocket_tts_mic_test_reference.wav"
    sf.write(str(path), pcm, sample_rate)
    print(f"Reference saved to {path} ({pcm.size / sample_rate:.1f}s)\n")
    return str(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--translate", choices=sorted(LANGUAGES), default="es")
    p.add_argument("--reference", default=None,
                   help="Wav file, named Pocket TTS voice, or hf:// path to clone "
                        "(default: record 8s from your mic)")
    p.add_argument("--reference-seconds", type=float, default=8.0)
    p.add_argument("--asr-model", default="small")
    p.add_argument("--asr-device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--pocket-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="auto stays on CPU; GPU support is unofficial (see module docstring)")
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

    print("Loading Whisper + Opus-MT …")
    translator = SpeechTranslator(args.asr_model, device=args.asr_device,
                                  threads=args.translate_threads, mt_size=args.mt_model,
                                  load_tts=False)
    log.info("speech recognition: whisper %s on %s", args.asr_model, translator.asr_device)
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

    translation = PeerTranslation(translator, "you", args.translate,
                                  out=on_pcm, backlog_ms=lambda: speaker.backlog_ms(key),
                                  speak_fn=voice.speak, turn_detector=turn_detector)

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
