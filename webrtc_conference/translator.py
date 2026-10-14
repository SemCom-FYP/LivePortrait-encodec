# coding: utf-8
"""
Receiver-side speech translation: a remote participant's voice, spoken back
in the listener's language instead of the original.

    decoded EnCodec PCM (24 kHz) -> 16 kHz -> Silero VAD -> Whisper (ASR)
        -> whole English sentences -> Opus-MT (English -> X) -> TTS
        -> 24 kHz -> speaker

Accuracy comes first: the translator only ever sees complete sentences, and
the recogniser always hears the sentence it is working on from its start —
see StreamingTranscriber. Cutting the audio into fixed pieces instead splits
sentences, and both models then guess at the half they never saw.

Deciding *when* a sentence has ended is itself two-layered. The baseline is a
fixed pause: `end_pause_s` of silence ends the sentence. On top of that,
`TurnDetector` wraps pipecat-ai's smart-turn
(https://github.com/pipecat-ai/smart-turn), a small Whisper-encoder
classifier that listens to a pause and judges whether the speaker's turn
actually ended there or they are just catching a breath. When it is confident
the turn ended, StreamingTranscriber finalizes the sentence right away
instead of waiting out the rest of the fixed pause; when it isn't, the fixed
pause remains the fallback, so a misjudged breath never loses a sentence.

Whisper and the translation model go on the GPU when there is one; on a CPU,
Whisper alone costs 1.3-2 s per reading. Speech synthesis runs on the CPU
(~0.1 s a sentence). Models are loaded once and shared; each remote peer gets
its own PeerTranslation with two threads, one recognising and one
translating, so a sentence is being translated while the next is heard.

Latency: no translator can say a sentence it has not heard, so the voice runs
about a sentence behind the speaker. It does not drift further: translations
often run longer than the original, so the speaking rate is chosen per
sentence to fit the time the original took and to work off any backlog.

Models live in pretrained_weights/translation/ and are fetched on first use:
an Opus-MT model is converted to an int8 CTranslate2 model once (~75 MB on
disk, ~230 MB for the larger `big` ones), and a Piper voice is ~60 MB.
"""

import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torchaudio.functional as AF

from .audio_codec import SAMPLE_RATE

log = logging.getLogger("translate")

ASR_RATE = 16000
MODEL_DIR = Path(__file__).resolve().parent.parent / "pretrained_weights" / "translation"
TURN_MODEL_DIR = Path(__file__).resolve().parent.parent / "pretrained_weights" / "turn_detection"
TURN_MODEL_REPO = "pipecat-ai/smart-turn-v3"
TURN_MODEL_FILE = "smart-turn-v3.2-cpu.onnx"

# Speaker-mixer key suffix for a peer's translated voice; it gets its own ring
# buffer so its backlog can be measured apart from the original's.
VOICE_SUFFIX = ":tr"
# Translated speech arrives a sentence at a time, so its ring buffer has to
# hold more than the original's 1.5 s or the start of a sentence is lost.
VOICE_BUFFER_S = 10.0
# Fastest the translated voice is sped up to keep pace; past ~1.35x Piper
# voices start to sound rushed.
MAX_SPEED = 1.35
# How much longer speech in the target language starts out assumed to run
# than the English it translates; learned per peer from there.
INITIAL_LENGTH_RATIO = 1.15

# This project's Language.code -> Kyutai Pocket TTS's own language names (one
# checkpoint per language; see https://github.com/kyutai-labs/pocket-tts).
POCKET_LANGUAGE = {"es": "spanish", "fr": "french", "de": "german", "it": "italian"}
# A bundled named voice (see https://huggingface.co/kyutai/tts-voices) used to
# clone Pocket TTS's speech when no --voice-reference is given.
DEFAULT_POCKET_VOICE = "alba"


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    mt_model: str           # Hugging Face id of an English -> X Opus-MT model
    voice: str              # Piper voice name
    mt_model_big: Optional[str] = None      # transformer-big version, if any


LANGUAGES = {lang.code: lang for lang in (
    Language("es", "Spanish", "Helsinki-NLP/opus-mt-en-es", "es_ES-davefx-medium",
             "Helsinki-NLP/opus-mt-tc-big-en-es"),
    Language("fr", "French", "Helsinki-NLP/opus-mt-en-fr", "fr_FR-siwis-medium",
             "Helsinki-NLP/opus-mt-tc-big-en-fr"),
    Language("de", "German", "Helsinki-NLP/opus-mt-en-de", "de_DE-thorsten-medium"),
    Language("it", "Italian", "Helsinki-NLP/opus-mt-en-it", "it_IT-paola-medium",
             "Helsinki-NLP/opus-mt-tc-big-en-it"),
)}


def _resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return pcm.astype(np.float32, copy=False)
    return AF.resample(torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))[None],
                       src, dst)[0].numpy()


# Whisper's training data leans heavily on YouTube, so on silence or
# background noise it tends to hallucinate one of a handful of outro
# phrases instead of staying quiet — "Thank you." above all. These drop at
# a much lower no_speech_prob than the general filter below requires: a
# genuine utterance of exactly one of these phrases is rare, and a wrongly
# dropped one just gets picked up again on the next reading if it really
# was speech. Matters doubly here, since an accepted hallucination gets fed
# back as every later reading's prompt (see `_context`), biasing Whisper to
# repeat it — which is how one false "Thank you." becomes ten.
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "thank you for watching",
    "please subscribe", "like and subscribe", "subscribe to my channel",
    "see you next time", "bye", "bye bye", "goodbye",
}


# ── models, shared by every peer ─────────────────────────────────────────────

class SpeechTranslator:
    """Whisper plus, per target language, an Opus-MT model and a Piper voice.

    All three are thread-safe to call concurrently (CTranslate2 and ONNX
    Runtime queue internally), so peers that talk at once are not serialised
    here; only lazy loading takes a lock.
    """

    def __init__(self, asr_model: Optional[str] = "small", source: str = "en",
                 device: str = "auto", threads: int = 4, mt_size: str = "base",
                 load_tts: bool = True):
        """`asr_model=None` skips loading Whisper entirely — for callers that
        bring their own recogniser (e.g. a Qwen3-ASR-based transcriber passed
        to PeerTranslation's `transcriber`) and only want this class for its
        Opus-MT translation and Piper voice.

        `load_tts=False` skips loading a Piper voice in `prepare()` as well —
        for callers that supply their own `speak_fn` to PeerTranslation (e.g.
        PocketTTSVoice below) and so never call this class's own `speak()`,
        and would otherwise pay for a Piper download/load that's never used."""
        import ctranslate2
        self.source = source
        self.threads = threads
        self.mt_size = mt_size
        self.load_tts = load_tts
        # Whisper encodes a padded 30 s window however short the utterance, so
        # on a CPU every call costs the same ~1.5-2 s for `small`; a GPU does
        # it in ~0.1 s, and the streaming recogniser reads every second.
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self.asr_device = device
        # Beam search finds noticeably better transcripts than greedy decoding
        # and is cheap on a GPU; on a CPU it would double an already long call.
        self.asr_beam = 5 if device == "cuda" else 1
        self.asr = None
        if asr_model is not None:
            from faster_whisper import WhisperModel
            opts = dict(device=device, cpu_threads=threads,
                        compute_type="float16" if device == "cuda" else "int8")
            try:
                # Cached copy first: asking the Hub on every launch can stall
                # the start of translation for as long as the network takes
                # to answer.
                self.asr = WhisperModel(asr_model, local_files_only=True, **opts)
            except Exception:
                log.info("downloading Whisper %s (one-time) …", asr_model)
                self.asr = WhisperModel(asr_model, **opts)
        self._mt: dict = {}
        self._voices: dict = {}
        self._load_lock = threading.Lock()

    def is_ready(self, code: str) -> bool:
        return code in self._mt and (not self.load_tts or code in self._voices)

    def prepare(self, code: str) -> None:
        """Load (downloading and converting on first use) one target language."""
        lang = LANGUAGES[code]
        with self._load_lock:
            if code not in self._mt:
                self._mt[code] = self._load_mt(lang)
            if self.load_tts and code not in self._voices:
                self._voices[code] = self._load_voice(lang)

    def _load_mt(self, lang: Language):
        import ctranslate2
        import transformers
        name = lang.mt_model_big if self.mt_size == "big" and lang.mt_model_big else lang.mt_model
        out = MODEL_DIR / f"{name.split('/')[-1]}-ct2"
        if not (out / "model.bin").exists():
            log.info("converting %s to CTranslate2 (one-time) …", name)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            # Loading a .bin checkpoint makes transformers fetch a safetensors
            # copy on a non-daemon thread — hundreds of MB we never use, which
            # on a slow link holds the app open after the window closes.
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
            # The tokenizer files travel with the model, so later launches
            # load everything from disk without touching the Hub.
            ctranslate2.converters.TransformersConverter(
                name, copy_files=["source.spm", "target.spm", "vocab.json",
                                  "tokenizer_config.json"],
            ).convert(str(out), quantization="int8", force=True)
        # Named outright: the folder's config.json is CTranslate2's, which
        # AutoTokenizer can't tell the tokenizer class from.
        tok = transformers.MarianTokenizer.from_pretrained(str(out))
        if self.asr_device == "cuda":
            mt = ctranslate2.Translator(str(out), device="cuda", compute_type="int8_float16")
        else:
            mt = ctranslate2.Translator(str(out), device="cpu", inter_threads=2,
                                        intra_threads=max(1, self.threads // 2))
        log.info("translation model: %s on %s", name, self.asr_device)
        return tok, mt

    def _load_voice(self, lang: Language):
        from piper import PiperVoice
        from piper.download_voices import download_voice
        path = MODEL_DIR / f"{lang.voice}.onnx"
        if not path.exists() or not Path(f"{path}.json").exists():
            log.info("downloading Piper voice %s (one-time) …", lang.voice)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            download_voice(lang.voice, MODEL_DIR)
        return PiperVoice.load(path)

    def words(self, audio16k: np.ndarray, prompt: Optional[str] = None) -> list:
        """(start s, end s, text) for each word Whisper hears, times relative
        to the start of `audio16k`; text keeps Whisper's leading space and
        punctuation."""
        segments, _ = self.asr.transcribe(
            audio16k, language=self.source, beam_size=self.asr_beam, temperature=0.0,
            condition_on_previous_text=False, word_timestamps=True,
            initial_prompt=prompt or None, vad_filter=False)
        out = []
        for s in segments:
            # Whisper invents text for noise and breath; its own confidence
            # numbers are enough to drop most of that, and a known
            # hallucination phrase (see HALLUCINATION_PHRASES) is dropped
            # at a lower bar still.
            text = s.text.strip().lower().translate(_STRIP)
            if ((s.no_speech_prob > 0.6 and s.avg_logprob < -0.7)
                    or s.avg_logprob < -1.2 or s.compression_ratio > 2.4
                    or (text in HALLUCINATION_PHRASES and s.no_speech_prob > 0.3)):
                continue
            out.extend((w.start, w.end, w.word) for w in s.words or ())
        return out

    def translate(self, text: str, code: str) -> str:
        tok, mt = self._mt[code]
        src = tok.convert_ids_to_tokens(tok.encode(text))
        hyp = mt.translate_batch([src], beam_size=4, max_decoding_length=256)[0].hypotheses[0]
        return tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).strip()

    def speak(self, text: str, code: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        """Float32 PCM at the conference rate, one piece per sentence."""
        from piper import SynthesisConfig
        voice = self._voices[code]
        cfg = SynthesisConfig(length_scale=1.0 / speed)
        for chunk in voice.synthesize(text, cfg):
            yield _resample(chunk.audio_float_array, chunk.sample_rate, SAMPLE_RATE)


class PocketTTSVoice:
    """Kyutai's Pocket TTS (https://github.com/kyutai-labs/pocket-tts) as a
    `speak_fn` for PeerTranslation, in place of SpeechTranslator's own Piper
    voice.

    Pocket TTS is a 100M-parameter model designed to run on a CPU (no GPU
    required, ~2-6x real-time on a laptop CPU), cloned from a short reference
    clip rather than shipped as one fixed voice per language — so unlike
    Piper there is one TTSModel *per target language* here (a separate
    checkpoint each), every one cloned to the same reference voice the first
    time that language is spoken.

    Needs the `pocket-tts` package (not in requirements.txt, optional):
        pip install pocket-tts
    `generate_audio_stream` is documented as not thread-safe for a given
    model instance; `_lock` serialises calls into any one of them, which is
    free here since PeerTranslation's own `_speak_loop` is already the only
    thread that ever calls `speak()` for a given peer — multiple peers each
    get their own PocketTTSVoice, so they are never serialised against each
    other.
    """

    def __init__(self, device: str = "auto", quantize: bool = False):
        # No GPU support is officially offered; "auto" stays on CPU, where
        # Pocket TTS is designed to run.
        self.device = "cpu" if device == "auto" else device
        self.quantize = quantize
        if quantize and self.device != "cpu":
            log.warning("quantize=True only works on CPU; ignoring it on %s", self.device)
            self.quantize = False
        self._lock = threading.Lock()
        self._reference = None
        self._models: dict = {}     # pocket-tts language name -> (TTSModel, voice_state)

    def clone(self, reference: str):
        """Remembers the reference clip (a wav path, a named Pocket TTS
        voice such as "alba", or an hf://kyutai/tts-voices/... path); actual
        cloning happens per language, lazily, the first time it's spoken."""
        self._reference = reference

    def _model_and_state(self, language_id: str):
        pocket_lang = POCKET_LANGUAGE.get(language_id)
        if pocket_lang is None:
            raise ValueError(f"pocket-tts has no model for language '{language_id}'; "
                             f"supported: {sorted(POCKET_LANGUAGE)}")
        if pocket_lang not in self._models:
            if self._reference is None:
                raise RuntimeError("clone() a reference voice before speak()")
            from pocket_tts import TTSModel
            log.info("loading pocket-tts %s on %s …", pocket_lang, self.device)
            t0 = time.time()
            model = TTSModel.load_model(language=pocket_lang, quantize=self.quantize)
            if self.device != "cpu":
                model.to(self.device)
            state = model.get_state_for_audio_prompt(self._reference)
            log.info("pocket-tts %s ready in %.1fs (native rate %d Hz)",
                     pocket_lang, time.time() - t0, model.sample_rate)
            self._models[pocket_lang] = (model, state)
        return self._models[pocket_lang]

    def speak(self, text: str, code: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        """Matches SpeechTranslator.speak's speak_fn(text, code, speed) contract."""
        model, state = self._model_and_state(code.lower())
        with self._lock:
            for chunk in model.generate_audio_stream(state, text):
                pcm = chunk.detach().to("cpu", dtype=torch.float32).numpy()
                if speed != 1.0:
                    # Pocket TTS has no rate control; resampling the result is
                    # a cheap way to speed up playback (pitch shifts slightly,
                    # the same trade-off translator.py's own callers accept
                    # elsewhere for a cloned voice).
                    n = max(1, int(pcm.size / speed))
                    pcm = np.interp(np.linspace(0, pcm.size - 1, n),
                                    np.arange(pcm.size), pcm).astype(np.float32)
                if model.sample_rate != SAMPLE_RATE:
                    pcm = _resample(pcm, model.sample_rate, SAMPLE_RATE)
                yield pcm


class TurnDetector:
    """Semantic turn-end detection: pipecat-ai's smart-turn
    (https://github.com/pipecat-ai/smart-turn), an 8M-parameter Whisper-encoder
    classifier that listens to speech ending in a pause and judges whether
    the speaker's turn is actually over, rather than guessing from how long
    the pause has lasted the way a fixed silence timeout does.

    One instance is shared by every peer's StreamingTranscriber (see
    `turn_detector` there); onnxruntime sessions serve concurrent `run()`
    calls from multiple threads, so no lock is needed here.
    """

    WINDOW_S = 8.0           # the model's fixed input window; see smart-turn's inference.py

    def __init__(self):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        path = TURN_MODEL_DIR / TURN_MODEL_FILE
        if not path.exists():
            import shutil
            from huggingface_hub import hf_hub_download
            log.info("downloading smart-turn %s (one-time) …", TURN_MODEL_FILE)
            TURN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download(TURN_MODEL_REPO, TURN_MODEL_FILE), path)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(path), sess_options=so)
        self._extractor = WhisperFeatureExtractor(chunk_length=int(self.WINDOW_S))
        self._max_samples = int(self.WINDOW_S * ASR_RATE)
        log.info("turn detection: smart-turn (%s)", TURN_MODEL_FILE)

    def complete_probability(self, audio16k: np.ndarray) -> float:
        """Probability in [0, 1] that `audio16k` — ending at the moment the
        speaker fell silent — is a finished turn rather than a mid-sentence
        breath. Only the last WINDOW_S seconds are seen, as the model was
        trained; shorter audio is zero-padded at the *start* so the real
        speech still ends where the model expects it."""
        n = self._max_samples
        audio = audio16k.astype(np.float32, copy=False)
        audio = audio[-n:] if audio.size > n else np.pad(audio, (n - audio.size, 0))
        inputs = self._extractor(audio, sampling_rate=ASR_RATE, return_tensors="np",
                                 padding="max_length", max_length=n,
                                 truncation=True, do_normalize=True)
        feats = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
        return self._session.run(None, {"input_features": feats})[0][0].item()


# ── utterance segmentation ───────────────────────────────────────────────────

class StreamingVAD:
    """Silero VAD (the copy bundled with faster-whisper) fed incrementally.

    The model is stateful — an LSTM plus 64 samples of the previous window —
    so that state is carried from call to call here instead of being reset
    for every chunk as faster-whisper's whole-file helper does.
    """

    WINDOW = 512                        # samples at 16 kHz = 32 ms
    CONTEXT = 64

    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self._session = get_vad_model().session
        self.reset()

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(self.CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def __call__(self, pcm16k: np.ndarray) -> list:
        """(window, speech probability) for every whole window now available."""
        buf = np.concatenate([self._pending, pcm16k.astype(np.float32, copy=False)])
        n = buf.size // self.WINDOW
        self._pending = buf[n * self.WINDOW:]
        if n == 0:
            return []
        wins = buf[:n * self.WINDOW].reshape(n, self.WINDOW)
        ctx = np.concatenate([self._ctx[None], wins[:-1, -self.CONTEXT:]], axis=0)
        self._ctx = wins[-1, -self.CONTEXT:].copy()
        out, self._h, self._c = self._session.run(
            None, {"input": np.concatenate([ctx, wins], axis=1), "h": self._h, "c": self._c})
        return list(zip(wins, np.asarray(out).reshape(-1)))


class StreamResampler:
    """24 kHz -> 16 kHz across chunk boundaries without seams.

    Resampling each 240 ms chunk on its own zero-pads both of its edges, which
    puts a small click into the speech every 240 ms. Here each call holds back
    the last few input samples and resamples them again, with real context on
    both sides, as the start of the next call.
    """

    def __init__(self, src: int = SAMPLE_RATE, dst: int = ASR_RATE, context: int = 48):
        g = np.gcd(src, dst)
        self.src, self.dst = src, dst
        self.up, self.down = dst // g, src // g
        self.ctx = -(-context // self.down) * self.down     # whole input periods
        self.reset()

    def reset(self):
        self._x = np.zeros(self.ctx, dtype=np.float32)

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._x, pcm.astype(np.float32, copy=False)])
        usable = (x.size - 2 * self.ctx) // self.down * self.down
        if usable <= 0:
            self._x = x
            return np.zeros(0, dtype=np.float32)
        y = _resample(x[:self.ctx + usable + self.ctx], self.src, self.dst)
        lo = self.ctx * self.up // self.down
        self._x = x[usable:]
        return y[lo:lo + usable * self.up // self.down]


SENTENCE_END = (".", "?", "!", "…")
CLAUSE_END = (",", ";", ":")
# Where a long sentence may be split (before these words). These always open
# a new clause, so each side still means what it said.
CLAUSE_WORDS = {"so", "because", "but", "which", "while", "although", "though",
                "whereas", "unless"}
# These often join two nouns instead ("Maria and John"), so a split before
# them is a last resort for a sentence that has run on far too long.
WEAK_CLAUSE_WORDS = {"and", "or", "then"}
_STRIP = str.maketrans("", "", ".,!?;:\"'()[]…-")


@dataclass
class Word:
    start: float                # stream time, s
    end: float
    text: str                   # as Whisper wrote it: leading space, punctuation
    final: bool = False         # a sentence ends here

    @property
    def key(self) -> str:
        return self.text.strip().lower().translate(_STRIP)


@dataclass
class Sentence:
    text: str
    start: float                # stream time of its first word, s
    end: float                  # ... and of the end of its last
    heard_at: float             # perf_counter() when its last word arrived


class StreamingTranscriber:
    """English speech in, finished sentences out — never split mid-sentence.

    Whisper re-reads a growing buffer of the speech in progress every second
    or so, and a word is accepted only once two readings in a row agree on it
    (LocalAgreement-2, as in whisper_streaming). The end of the buffer is
    where Whisper guesses; a guess it repeats once more audio has arrived
    after it is almost always right.

    Accepted words are released as whole sentences: at Whisper's own sentence
    punctuation, trusted only when it wrote it with more speech after it (at
    the very end of the audio Whisper puts a full stop whatever comes next),
    or when the speaker has stopped for `end_pause_s`. A pause shorter than
    that stays inside the buffer, so a sentence that pauses for breath is
    still read as one. Only once a sentence is released is the audio before
    it dropped: the recogniser always hears the current sentence from its
    first word.

    A sentence still going after `long_sentence_s` gives up the clauses it
    has finished — split at a comma or before "so", "because", "but", … where
    each side keeps its meaning — so its translation can start while the
    speaker is still talking. There is no fixed time limit past that: with no
    clause boundary to split at, it keeps growing until the speaker actually
    pauses, *unless* `turn_detector` is given, in which case it is asked
    whether the speech so far already sounds finished; a confident "yes"
    cuts it there — now trying "and"/"or"/"then" too, which otherwise are a
    last resort since they as often join two nouns — rather than running on
    indefinitely.

    When given a `turn_detector` (see TurnDetector), each pause is also asked
    — once, after `turn_check_s` of silence — whether it sounds like the end
    of the speaker's turn. A confident "yes" finalizes the sentence right
    then, before the full `end_pause_s` has elapsed; anything less confident
    leaves `end_pause_s` as the fallback, same as with no detector at all.
    """

    START_TH, END_TH = 0.5, 0.35        # VAD speech probability
    PAD = 3                             # windows of silence kept around speech
    MIN_CLAUSE_S = 2.5                  # shortest piece a split may leave in front

    def __init__(self, asr: SpeechTranslator, step_s: float = 1.0,
                 end_pause_s: float = 0.8, long_sentence_s: float = 6.0,
                 max_buffer_s: float = 25.0,
                 turn_detector: Optional[TurnDetector] = None,
                 turn_check_s: float = 0.3, turn_complete_th: float = 0.5):
        self.asr = asr
        self.vad = StreamingVAD()
        self.win_s = StreamingVAD.WINDOW / ASR_RATE
        self.step_s = step_s
        self.end_pause = round(end_pause_s / self.win_s)
        self.long_sentence_s = long_sentence_s
        self.max_buffer = round(max_buffer_s / self.win_s)
        self.turn_detector = turn_detector
        self.turn_check = round(turn_check_s / self.win_s)
        self.turn_complete_th = turn_complete_th
        self.asr_ms = 0.0
        self.reset()

    def reset(self):
        self.vad.reset()
        self._t = 0.0                   # stream time at the end of audio seen
        self._arrivals: deque = deque(maxlen=512)     # (stream t, perf_counter)
        self._preroll: deque = deque(maxlen=self.PAD)
        self._wins: list = []           # the speech in progress, 32 ms windows
        self._buf_t0 = 0.0              # stream time of _wins[0]
        self._active = False
        self._silence = 0
        self._turn_checked = False      # smart-turn already asked about this pause
        self._since_read = 0.0
        self._hyp: list = []            # last reading, past what is accepted
        self._pending: list = []        # accepted, not yet a whole sentence
        self._tail: list = []           # last few accepted words, for de-duping
        self._last_end = 0.0            # end of the last accepted word
        self._context = ""              # released text, Whisper's prompt

    def push(self, pcm16k: np.ndarray) -> list:
        """Feed audio; returns the sentences it completed."""
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
                    self._silence, self._since_read = 0, 0.0
                continue
            self._wins.append(win)
            self._since_read += self.win_s
            if p < self.END_TH:
                if self._silence == 0:
                    self._turn_checked = False
                self._silence += 1
            else:
                self._silence = 0
            if self._silence >= self.end_pause:
                # The speaker has stopped: one last reading of everything,
                # without the trailing silence, and all of it goes out.
                out += self._finish_utterance()
            elif (self.turn_detector is not None and not self._turn_checked
                    and self._silence >= self.turn_check):
                self._turn_checked = True
                if self._turn_sounds_finished():
                    out += self._finish_utterance()
        if (self._active and self._since_read >= self.step_s
                and len(self._wins) * self.win_s >= 1.0):
            out += self._read(final=False)
        return out

    def _finish_utterance(self) -> list:
        """The speaker has stopped (confirmed by a fixed pause, or sooner by
        smart-turn): one last reading of everything but the trailing
        silence, and the speech buffer resets for the next utterance."""
        del self._wins[len(self._wins) - self._silence + self.PAD:]
        out = self._read(final=True)
        self._active = False
        self._wins, self._hyp = [], []
        self._preroll.clear()
        return out

    def _turn_sounds_finished(self) -> bool:
        """Asks `turn_detector` about the speech buffered so far — usually
        ending in the pause currently in progress, or still running on with
        no pause at all (see `_release`'s use of this for an overlong
        sentence). False (never finalize early) on any inference failure,
        leaving the fixed pause, or an eventual clause boundary, as the
        fallback."""
        try:
            prob = self.turn_detector.complete_probability(np.concatenate(self._wins))
        except Exception as exc:
            log.warning("smart-turn inference failed: %s", exc)
            return False
        return prob >= self.turn_complete_th

    def _read(self, final: bool) -> list:
        t0 = time.perf_counter()
        raw = self.asr.words(np.concatenate(self._wins), self._context[-200:])
        self.asr_ms = (time.perf_counter() - t0) * 1000
        self._since_read = 0.0
        # Readings of a sentence in progress run longer than one step on a
        # slow device; reading less often keeps it from falling behind.
        self.step_s = max(1.0, 1.5 * self.asr_ms / 1000)

        words = [Word(self._buf_t0 + s, self._buf_t0 + e, txt) for s, e, txt in raw]
        new = [w for w in words if w.start > self._last_end - 0.1]
        # The buffer still holds accepted words; drop any the reading repeats.
        if new and self._tail and abs(new[0].start - self._last_end) < 1.0:
            for n in range(min(5, len(new), len(self._tail)), 0, -1):
                if [w.key for w in self._tail[-n:]] == [w.key for w in new[:n]]:
                    new = new[n:]
                    break
        if final:
            accepted, self._hyp = new, []
        else:
            k = 0
            while k < min(len(new), len(self._hyp)) and new[k].key == self._hyp[k].key:
                k += 1
            accepted, self._hyp = new[:k], new[k:]
        for i, w in enumerate(accepted):
            w.final = w.text.rstrip().endswith(SENTENCE_END) and (final or i + 1 < len(new))
        if accepted:
            self._pending += accepted
            self._tail = (self._tail + accepted)[-5:]
            self._last_end = accepted[-1].end

        out = self._release(everything=final)
        if out and not final:
            self._trim(out[-1].end)
        elif len(self._wins) > self.max_buffer:
            self._trim(self._pending[0].start if self._pending else self._last_end)
        return out

    def _release(self, everything: bool) -> list:
        out = []
        while True:
            i = next((i for i, w in enumerate(self._pending) if w.final), None)
            if i is None:
                break
            out.append(self._sentence(self._pending[:i + 1]))
            self._pending = self._pending[i + 1:]
        cut = None
        if self._pending and everything:
            cut = len(self._pending)
        elif self._pending:
            span = self._pending[-1].end - self._pending[0].start
            if span > self.long_sentence_s:
                cut = self._clause_cut(self._pending, CLAUSE_WORDS)
                if cut is None and self.turn_detector is not None and self._turn_sounds_finished():
                    # No grammatical boundary, but smart-turn says this run-on
                    # sentence already sounds finished: cut it now, trying the
                    # weaker joiners too, rather than let it grow indefinitely.
                    cut = (self._clause_cut(self._pending, CLAUSE_WORDS | WEAK_CLAUSE_WORDS)
                           or len(self._pending))
        if cut:
            out.append(self._sentence(self._pending[:cut]))
            self._pending = self._pending[cut:]
        for s in out:
            self._context = (self._context + " " + s.text)[-400:]
        return out

    def _clause_cut(self, words: list, openers: set) -> Optional[int]:
        """Index to split `words` at: the latest clause boundary that leaves
        at least MIN_CLAUSE_S in front, or None."""
        for i in range(len(words) - 1, 0, -1):
            if words[i].key in openers:
                cut = i                                 # before the opener
            elif words[i - 1].text.rstrip().endswith(CLAUSE_END):
                cut = i                                 # after the comma
            else:
                continue
            if words[cut - 1].end - words[0].start >= self.MIN_CLAUSE_S:
                return cut
        return None

    def _sentence(self, words: list) -> Sentence:
        end = words[-1].end
        heard = next((wall for t, wall in self._arrivals if t >= end), time.perf_counter())
        # A full stop Whisper put at a pause mid-sentence, before the speaker
        # went on in lower case, would tell the translator the sentence ended.
        parts = [w.text for w in words]
        for i in range(len(parts) - 1):
            nxt = parts[i + 1].strip()
            if parts[i].endswith(".") and nxt[:1].islower():
                parts[i] = parts[i][:-1]
        return Sentence("".join(parts).strip(), words[0].start, end, heard)

    def _trim(self, t: float):
        """Drop buffered audio before stream time `t`, never into a word
        that is still waiting to be accepted."""
        nxt = self._pending[0].start if self._pending else (
            self._hyp[0].start if self._hyp else None)
        if nxt is not None:
            t = min(t, nxt - 0.05) if nxt < t else (t + nxt) / 2
        n = int((t - self._buf_t0) / self.win_s)
        n = max(0, min(n, len(self._wins) - 1))
        del self._wins[:n]
        self._buf_t0 += n * self.win_s


# ── one remote peer ──────────────────────────────────────────────────────────

class PeerTranslation:
    """Translates one remote participant's voice into `target`.

    `feed` is called from the shared audio decode loop and never blocks it.
    Finished speech goes to `out(pcm)`; `backlog_ms()` reports how much of it
    is still waiting in the speaker, which is what the pacing keys off.
    """

    def __init__(self, translator: SpeechTranslator, name: str, target: Optional[str],
                 out: Callable[[np.ndarray], None], backlog_ms: Callable[[], float],
                 speak_fn: Optional[Callable[[str, str, float], Iterator[np.ndarray]]] = None,
                 turn_detector: Optional[TurnDetector] = None,
                 transcriber: Optional[object] = None):
        """`speak_fn(text, language_code, speed) -> iterator of float32 PCM at
        SAMPLE_RATE` replaces the translator's own (Piper) voice when given —
        e.g. a voice-cloning backend such as Chatterbox. Defaults to
        `translator.speak`, so existing callers are unaffected.

        `turn_detector` (see TurnDetector), if given, lets the recogniser
        finalize a sentence as soon as a pause sounds like the end of the
        speaker's turn, instead of always waiting out the fixed pause. A
        single instance is meant to be shared across every peer's
        PeerTranslation.

        `transcriber`, if given, replaces the default StreamingTranscriber
        (Whisper-based) recogniser — e.g. a recogniser built around a model
        without word timestamps, such as Qwen3-ASR. It must offer the same
        `push(pcm16k) -> list[Sentence]` / `reset()` contract as
        StreamingTranscriber; `turn_detector` is then its caller's concern,
        not built here."""
        self.tr, self.name, self.target = translator, name, target
        self._out, self._backlog_ms = out, backlog_ms
        self._speak_fn = speak_fn or translator.speak
        self._in: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
        self._sentences: "queue.Queue[tuple[int, Sentence]]" = queue.Queue()
        # Separate from _sentences so a slow voice (speak_fn) backs up only
        # itself: translation of what's said next must not wait on synthesis
        # of what was said before. With Piper (near-instant) this queue never
        # really builds up; with a slower cloned voice it does, visibly, and
        # that backlog is what lag_ms then reports.
        self._to_speak: "queue.Queue[tuple[int, str, Sentence, str]]" = queue.Queue()
        self._resampler = StreamResampler()
        self._asr = transcriber or StreamingTranscriber(translator, turn_detector=turn_detector)
        self._reset = False             # set from outside, acted on by the recogniser
        self._gen = 0                   # bumped on retarget; stale work is dropped
        # Translated speech length / original length at normal speed, learned
        # as the call goes; it sets the speaking rate (see _speak).
        self._length_ratio = INITIAL_LENGTH_RATIO
        self.running = True

        # Read by the UI.
        self.lag_ms = 0.0               # sentence spoken -> its translation audible
        self.last_text = ""

        threading.Thread(target=self._asr_loop, daemon=True,
                         name=f"tr-asr-{name}").start()
        threading.Thread(target=self._translate_loop, daemon=True,
                         name=f"tr-mt-{name}").start()
        threading.Thread(target=self._speak_loop, daemon=True,
                         name=f"tr-tts-{name}").start()

    def feed(self, pcm24k: np.ndarray) -> None:
        try:
            self._in.put_nowait(pcm24k)
        except queue.Full:
            pass

    def set_target(self, code: Optional[str]) -> None:
        """Switch language, or pause with None. Speech already heard in the
        old setting is dropped rather than finished."""
        self.target = code
        self._length_ratio = INITIAL_LENGTH_RATIO
        self._gen += 1
        self._drain(self._in)
        self._drain(self._sentences)
        self._drain(self._to_speak)
        self._reset = True

    def close(self) -> None:
        self.running = False

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _asr_loop(self):
        while self.running:
            try:
                chunks = [self._in.get(timeout=0.25)]
            except queue.Empty:
                continue
            # Whatever piled up while Whisper was busy goes in as one push,
            # so the recogniser catches up in a single reading.
            while True:
                try:
                    chunks.append(self._in.get_nowait())
                except queue.Empty:
                    break
            if self._reset:
                self._reset = False
                self._resampler.reset()
                self._asr.reset()
            gen = self._gen
            try:
                sentences = self._asr.push(self._resampler(np.concatenate(chunks)))
            except Exception as exc:
                log.warning("[%s] speech recognition failed: %s", self.name, exc)
                continue
            for s in sentences:
                self._sentences.put((gen, s))

    def _translate_loop(self):
        while self.running:
            try:
                gen, sentence = self._sentences.get(timeout=0.25)
            except queue.Empty:
                continue
            code = self.target
            if gen != self._gen or code is None:
                continue
            try:
                translated = self.tr.translate(sentence.text, code)
            except Exception as exc:
                log.warning("[%s] translation failed: %s", self.name, exc)
                continue
            if translated and gen == self._gen:
                self._to_speak.put((gen, code, sentence, translated))

    def _speak_loop(self):
        """Its own thread so a voice slower than real time (a cloned voice
        can be; Piper never was) backs up this queue instead of stalling
        _translate_loop — otherwise every sentence's translation would wait
        on every earlier sentence's speech finishing, compounding delay
        across the whole conversation rather than costing only itself."""
        while self.running:
            try:
                gen, code, sentence, translated = self._to_speak.get(timeout=0.25)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                self._speak(gen, code, sentence, translated)
            except Exception as exc:
                log.warning("[%s] speech synthesis failed: %s", self.name, exc)

    def _speak(self, gen: int, code: str, sentence: Sentence, translated: str):
        # Pick the speaking rate before synthesising: aim to say this in the
        # time the original took, less half of whatever is still queued. Left
        # at normal speed, Spanish or French runs ~15-20 % longer than the
        # English and the lag grows with every sentence.
        src_s = max(0.5, sentence.end - sentence.start)
        backlog = self._backlog_ms()
        natural_s = self._length_ratio * src_s
        budget_s = max(0.75 * src_s, src_s - 0.5 * backlog / 1000)
        speed = min(MAX_SPEED, max(1.0, natural_s / budget_s))
        spoken = 0
        for pcm in self._speak_fn(translated, code, speed):
            if gen != self._gen:
                return
            self._out(pcm)
            if not spoken:
                self.lag_ms = (time.perf_counter() - sentence.heard_at) * 1000 + backlog
            spoken += pcm.size
        if spoken:
            ratio = spoken / SAMPLE_RATE * speed / src_s
            self._length_ratio += 0.3 * (min(2.0, max(0.5, ratio)) - self._length_ratio)
        self.last_text = translated
        log.info("[%s] %s -> %s  (%.0f ms after it was said)",
                 self.name, sentence.text, translated, self.lag_ms)
