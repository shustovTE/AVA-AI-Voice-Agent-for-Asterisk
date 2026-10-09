"""Smart Turn v3 as the semantic layer above Silero VAD.

A voice-activity detector knows that the caller's sound stopped, not whether
their thought did: «я хочу…» followed by a pause reads exactly like «да».
Smart Turn (pipecat-ai/smart-turn, BSD-2) is an audio-native turn detector, a
Whisper Tiny encoder with a linear head (8M parameters, 8 MB int8 ONNX) that
scores the caller's own audio for whether the turn is complete, from prosody
and content rather than from a transcript. It supports 23 languages, Russian
among them, and runs in tens of milliseconds on one CPU core.

The model is run only when Silero VAD reports the caller quiet, on up to the
last 8 seconds of the caller's turn at 16 kHz mono, zero-padded at the front
when shorter. ``complete`` lets the turn go as usual; ``incomplete`` holds it
a while longer in case the caller goes on, exactly as the Pipecat reference
integration does. Feature extraction is a numpy re-implementation of the
Whisper log-mel front end (``transformers.WhisperFeatureExtractor``,
Apache-2.0; the same math is vendored in Pipecat), validated against it, so
the engine needs nothing beyond numpy and onnxruntime.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import structlog

from . import model_fetch

try:  # pragma: no cover - exercised through the availability flag
    import onnxruntime as _onnxruntime  # pyright: ignore[reportMissingImports]

    ONNXRUNTIME_AVAILABLE = True
except ImportError:  # pragma: no cover
    _onnxruntime = None  # type: ignore[assignment]
    ONNXRUNTIME_AVAILABLE = False

logger = structlog.get_logger(__name__)

# The pinned release: the int8 CPU build of Smart Turn v3.2 from the
# pipecat-ai/smart-turn-v3 repository on Hugging Face.
SMART_TURN_VERSION = "3.2"
SMART_TURN_MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
)
SMART_TURN_MODEL_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
SMART_TURN_MODEL_BYTES = 8679182
DEFAULT_MODEL_PATH = "models/turn/smart-turn-v3.2-cpu.onnx"
FETCH_SCRIPT = "scripts/fetch_smart_turn.sh"

MODEL_SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = MODEL_SAMPLE_RATE * WINDOW_SECONDS

# Whisper front end: 25 ms frames every 10 ms, 80 mel bins, 800 frames per window.
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_FRAMES = WINDOW_SAMPLES // HOP_LENGTH
MEL_FLOOR = 1e-10
NORMALIZE_EPS = 1e-7

DEFAULT_THRESHOLD = 0.5
# How long an incomplete verdict may hold the turn beyond the detector's stop
# (Pipecat's SmartTurnParams.stop_secs), how much of the trailing silence the
# model is shown, and how long the turn waits for a verdict at most.
DEFAULT_INCOMPLETE_HOLD_MS = 3000
DEFAULT_TRAILING_SILENCE_MS = 200
DEFAULT_TIMEOUT_MS = 500


class SmartTurnError(RuntimeError):
    """The model cannot be used; the message says why and what to do."""


# --- Whisper log-mel front end -------------------------------------------------


def _hz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    log_region = freq >= 1000.0
    mels = np.where(log_region, 15.0 + np.log(np.maximum(freq, 1e-9) / 1000.0) * (27.0 / np.log(6.4)), mels)
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    log_region = mels >= 15.0
    freq = np.where(log_region, 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels - 15.0)), freq)
    return freq


def mel_filterbank(
    n_bins: int = N_FFT // 2 + 1,
    n_mels: int = N_MELS,
    min_hz: float = 0.0,
    max_hz: float = MODEL_SAMPLE_RATE / 2.0,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> np.ndarray:
    """Slaney-normalized triangular filters, shape ``(n_bins, n_mels)``."""
    mel_points = np.linspace(_hz_to_mel_slaney(np.array(min_hz))[()], _hz_to_mel_slaney(np.array(max_hz))[()], n_mels + 2)
    hz_points = _mel_to_hz_slaney(mel_points)
    fft_hz = np.linspace(0.0, sample_rate // 2, n_bins)
    diff = np.diff(hz_points)
    slopes = hz_points[np.newaxis, :] - fft_hz[:, np.newaxis]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= (2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels]))[np.newaxis, :]
    return filters


_HANN = np.hanning(N_FFT + 1)[:-1]  # periodic, as torch.hann_window
_MEL_FILTERS = mel_filterbank()


def log_mel_features(audio: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Whisper log-mel features for one 8 s window, shape ``(80, 800)`` float32.

    Mirrors ``WhisperFeatureExtractor(chunk_length=8)`` called with
    ``padding="max_length"``, ``do_normalize=True``: the waveform is
    zero-mean unit-variance normalized as a whole (padding included), framed
    with reflect padding, projected onto 80 Slaney mels, log10-compressed,
    clamped to 8 dB below its peak and rescaled to roughly [-1, 1].
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < WINDOW_SAMPLES:
        x = np.pad(x, (WINDOW_SAMPLES - x.size, 0))
    elif x.size > WINDOW_SAMPLES:
        x = x[-WINDOW_SAMPLES:]
    if normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + NORMALIZE_EPS)
    padded = np.pad(x.astype(np.float64), (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spectrum = np.fft.rfft(frames * _HANN, axis=-1)
    power = (np.abs(spectrum) ** 2).T  # (n_bins, n_frames)
    mel = np.maximum(MEL_FLOOR, _MEL_FILTERS.T @ power)
    log_spec = np.log10(mel)[:, :N_FRAMES]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


# --- audio window --------------------------------------------------------------


def _lowpass_taps(taps: int = 63, cutoff: float = 0.25, beta: float = 8.0) -> np.ndarray:
    """Windowed-sinc low-pass; ``cutoff`` in cycles per sample of the output rate."""
    n = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.kaiser(taps, beta)
    return h / h.sum()


_UPSAMPLE_2X = _lowpass_taps() * 2.0


def upsample_2x(samples: np.ndarray) -> np.ndarray:
    """8 kHz to 16 kHz by zero-stuffing and a low-pass at 4 kHz.

    Telephony audio carries nothing above 4 kHz; a proper interpolation keeps
    it that way, where linear interpolation would fold images of the speech
    band into the octave the model also looks at.
    """
    x = np.asarray(samples, dtype=np.float32)
    stuffed = np.zeros(x.size * 2, dtype=np.float64)
    stuffed[::2] = x
    return np.convolve(stuffed, _UPSAMPLE_2X, mode="same").astype(np.float32)


def prepare_window(pcm16: bytes, sample_rate: int) -> np.ndarray:
    """Turn wire-rate PCM16 into the model's 8 s float window at 16 kHz.

    Longer audio keeps its end; shorter audio is zero-padded at the front so
    the caller's last words sit at the end of the window, as the model was
    trained.
    """
    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    rate = int(sample_rate)
    if rate == 8000:
        samples = upsample_2x(samples)
    elif rate != MODEL_SAMPLE_RATE and samples.size:
        target = int(round(samples.size * MODEL_SAMPLE_RATE / rate))
        samples = np.interp(
            np.linspace(0.0, samples.size - 1, target),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)
    if samples.size > WINDOW_SAMPLES:
        samples = samples[-WINDOW_SAMPLES:]
    elif samples.size < WINDOW_SAMPLES:
        samples = np.pad(samples, (WINDOW_SAMPLES - samples.size, 0))
    return samples


class TurnAudioBuffer:
    """The caller's recent audio at the wire rate, bounded to one window."""

    def __init__(self, max_seconds: float = WINDOW_SECONDS + 1.0) -> None:
        self.max_seconds = float(max_seconds)
        self.sample_rate: Optional[int] = None
        self._chunks: Deque[bytes] = deque()
        self._bytes = 0

    def clear(self) -> None:
        self._chunks.clear()
        self._bytes = 0

    def append(self, pcm16: bytes, sample_rate: int) -> None:
        rate = int(sample_rate)
        if rate != self.sample_rate:
            self.clear()
            self.sample_rate = rate
        if not pcm16:
            return
        self._chunks.append(bytes(pcm16))
        self._bytes += len(pcm16)
        limit = int(self.max_seconds * rate) * 2
        while self._bytes > limit and len(self._chunks) > 1:
            self._bytes -= len(self._chunks.popleft())

    @property
    def duration_ms(self) -> float:
        if not self.sample_rate:
            return 0.0
        return self._bytes / 2.0 / self.sample_rate * 1000.0

    def snapshot(self, *, trim_trailing_ms: float = 0.0) -> Tuple[bytes, int]:
        """The buffered audio minus the last ``trim_trailing_ms``, and its rate."""
        data = b"".join(self._chunks)
        rate = int(self.sample_rate or MODEL_SAMPLE_RATE)
        trim = int(max(0.0, trim_trailing_ms) * rate / 1000.0) * 2
        if trim and trim < len(data):
            data = data[:-trim]
        return data, rate


# --- the model -------------------------------------------------------------------


class SmartTurnModel:
    """One loaded ONNX session, shared by every call; stateless per request."""

    def __init__(self, session: Any, path: str) -> None:
        self._session = session
        self.path = path
        names = {inp.name for inp in session.get_inputs()}
        if "input_features" not in names:
            raise SmartTurnError(f"{path} is not a Smart Turn v3 graph (inputs {sorted(names)})")
        self._lock = threading.Lock()

    def probability(self, window: np.ndarray) -> float:
        """Probability that the turn whose audio ends the window is complete."""
        features = log_mel_features(window)[np.newaxis, :, :]
        with self._lock:
            outputs = self._session.run(None, {"input_features": features})
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def predict(self, pcm16: bytes, sample_rate: int) -> Dict[str, Any]:
        """Score wire-rate PCM16; returns probability, audio length and timing."""
        started = time.perf_counter()
        window = prepare_window(pcm16, sample_rate)
        probability = self.probability(window)
        return {
            "probability": probability,
            "audio_ms": len(pcm16) / 2.0 / max(1, int(sample_rate)) * 1000.0,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }


def ensure_model_file(
    path: str = DEFAULT_MODEL_PATH,
    *,
    auto_download: bool = True,
    timeout: float = 120.0,
    opener: Any = None,
) -> str:
    return model_fetch.ensure_file(
        path,
        url=SMART_TURN_MODEL_URL,
        sha256=SMART_TURN_MODEL_SHA256,
        version=SMART_TURN_VERSION,
        auto_download=auto_download,
        timeout=timeout,
        opener=opener,
        fetch_hint=f"run {FETCH_SCRIPT} or set vad.smart_turn_auto_download: true",
        error=SmartTurnError,
        label="Smart Turn model",
    )


_MODEL_CACHE: Dict[str, SmartTurnModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model(path: str, *, threads: int = 1) -> SmartTurnModel:
    """Load (once per path) the ONNX session on the CPU provider."""
    if not ONNXRUNTIME_AVAILABLE:
        raise SmartTurnError(
            "onnxruntime is not installed; add it to the engine image (requirements.txt) to use Smart Turn"
        )
    key = os.path.abspath(path)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        options = _onnxruntime.SessionOptions()
        options.execution_mode = _onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max(1, int(threads))
        options.graph_optimization_level = _onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        try:
            session = _onnxruntime.InferenceSession(
                key, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise SmartTurnError(f"cannot load the Smart Turn model at {path}: {exc}") from exc
        model = SmartTurnModel(session, key)
        _MODEL_CACHE[key] = model
        return model


def describe() -> Dict[str, Any]:
    return {
        "version": SMART_TURN_VERSION,
        "url": SMART_TURN_MODEL_URL,
        "sha256": SMART_TURN_MODEL_SHA256,
        "bytes": SMART_TURN_MODEL_BYTES,
        "window_seconds": WINDOW_SECONDS,
        "sample_rate": MODEL_SAMPLE_RATE,
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
    }
