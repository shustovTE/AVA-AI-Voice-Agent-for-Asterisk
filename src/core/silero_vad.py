"""Silero VAD as the engine's own caller-speech detector.

Asterisk ``TALK_DETECT`` and the WebRTC/energy heuristics all decide "speech"
from signal energy, so breathing, line noise or a television count as the
caller while a quiet trailing syllable counts as silence. Silero VAD is a small
neural network (about 2 MB as ONNX) that scores every 32 ms of audio with a
speech probability, natively at 8 or 16 kHz and in well under a millisecond on
one CPU core. Run inside the engine on the very frames that reach the
recognizer, it reports the start and the end of caller speech both robustly and
exactly timed, so one detector can drive barge-in, the inactivity watchdog and
the end of a turn, and the recognizer can be told to finalize the moment the
caller stops.

The model file is not bundled. The engine loads it from
``vad.silero_model_path`` and, when allowed, fetches the pinned release into
that path once, verifying its SHA-256. ``onnxruntime`` is the only runtime
dependency; the ``silero-vad`` package itself (which pulls in torch) is not
needed.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# The pinned release. The same file ships inside the silero-vad 6.2.1 wheel
# (silero_vad/data/silero_vad.onnx); the sha256 below is that file's.
SILERO_VAD_VERSION = "6.2.1"
SILERO_VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"v{SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_VAD_MODEL_BYTES = 2327524
DEFAULT_MODEL_PATH = "models/vad/silero_vad.onnx"
FETCH_SCRIPT = "scripts/fetch_silero_vad.sh"

# The graph scores one chunk at a time and carries a short context of the
# previous chunk in front of it, exactly as the reference wrapper does.
CHUNK_SAMPLES: Dict[int, int] = {8000: 256, 16000: 512}
CONTEXT_SAMPLES: Dict[int, int] = {8000: 32, 16000: 64}
SUPPORTED_SAMPLE_RATES: Tuple[int, ...] = tuple(CHUNK_SAMPLES)
CHUNK_MS = 32.0
STATE_SHAPE = (2, 1, 128)

# Silero's own offset margin: speech ends below ``threshold - 0.15``.
STOP_THRESHOLD_MARGIN = 0.15

DEFAULT_THRESHOLD = 0.5
DEFAULT_START_MS = 96
DEFAULT_STOP_MS = 300


class SileroVadError(RuntimeError):
    """The model cannot be used; the message says why and what to do."""


class SileroVadModel:
    """One loaded ONNX session, shared by every call.

    Inference state travels in and out of the graph explicitly, so a single
    session serves any number of concurrent streams; each stream keeps its own
    state and context (:class:`SileroVadStream`).
    """

    def __init__(self, session: Any, path: str) -> None:
        self._session = session
        self.path = path
        self._input_names = {inp.name for inp in session.get_inputs()}
        missing = {"input", "state", "sr"} - self._input_names
        if missing:
            raise SileroVadError(
                f"{path} is not a Silero VAD v5/v6 graph (missing inputs {sorted(missing)})"
            )

    def run(
        self, samples: np.ndarray, state: np.ndarray, sample_rate: int
    ) -> Tuple[float, np.ndarray]:
        """Score one context+chunk window; returns (probability, next state)."""
        outputs = self._session.run(
            None,
            {
                "input": samples,
                "state": state,
                "sr": np.array(sample_rate, dtype=np.int64),
            },
        )
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        return probability, np.asarray(outputs[1], dtype=np.float32)


class SileroVadStream:
    """Per-call chunker and recurrent state on top of a shared model."""

    def __init__(self, model: SileroVadModel, sample_rate: int) -> None:
        if sample_rate not in CHUNK_SAMPLES:
            raise ValueError(
                f"Silero VAD runs at {SUPPORTED_SAMPLE_RATES} Hz, not {sample_rate}"
            )
        self.model = model
        self.sample_rate = int(sample_rate)
        self.chunk_samples = CHUNK_SAMPLES[self.sample_rate]
        self.context_samples = CONTEXT_SAMPLES[self.sample_rate]
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(self.context_samples, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(self.context_samples, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def feed(self, pcm16: bytes) -> List[float]:
        """Consume little-endian PCM16 and return one probability per full chunk."""
        if not pcm16:
            return []
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        buffer = np.concatenate([self._pending, samples]) if len(self._pending) else samples
        probabilities: List[float] = []
        size = self.chunk_samples
        offset = 0
        while len(buffer) - offset >= size:
            chunk = buffer[offset : offset + size]
            window = np.concatenate([self._context, chunk])[np.newaxis, :]
            probability, self._state = self.model.run(
                np.ascontiguousarray(window, dtype=np.float32),
                self._state,
                self.sample_rate,
            )
            self._context = chunk[-self.context_samples :].copy()
            probabilities.append(probability)
            offset += size
        self._pending = buffer[offset:].copy()
        return probabilities


class SpeechSegmenter:
    """Turn per-chunk probabilities into ``start`` and ``stop`` events.

    Speech starts once ``start_ms`` of consecutive chunks score at or above
    ``threshold`` and stops once ``stop_ms`` have passed since the last chunk
    that did, measured from the first chunk below ``stop_threshold`` (Silero's
    own hysteresis: chunks between the two thresholds neither end speech nor
    restart it).
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        stop_threshold: Optional[float] = None,
        start_ms: float = DEFAULT_START_MS,
        stop_ms: float = DEFAULT_STOP_MS,
        chunk_ms: float = CHUNK_MS,
    ) -> None:
        self.threshold = min(1.0, max(0.0, float(threshold)))
        if stop_threshold is None:
            stop_threshold = self.threshold - STOP_THRESHOLD_MARGIN
        self.stop_threshold = min(self.threshold, max(0.0, float(stop_threshold)))
        self.start_ms = max(0.0, float(start_ms))
        self.stop_ms = max(0.0, float(stop_ms))
        self.chunk_ms = float(chunk_ms)
        self.talking = False
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.segment_ms = 0.0

    def reset(self) -> None:
        self.talking = False
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.segment_ms = 0.0

    def update(self, probability: float) -> Optional[str]:
        if not self.talking:
            if probability >= self.threshold:
                self.speech_ms += self.chunk_ms
                if self.speech_ms >= self.start_ms:
                    self.talking = True
                    self.silence_ms = 0.0
                    self.segment_ms = self.speech_ms
                    return "start"
            elif probability < self.stop_threshold:
                self.speech_ms = 0.0
            return None
        self.segment_ms += self.chunk_ms
        if probability >= self.threshold:
            self.silence_ms = 0.0
        elif probability < self.stop_threshold or self.silence_ms > 0.0:
            self.silence_ms += self.chunk_ms
            if self.silence_ms >= self.stop_ms:
                self.talking = False
                self.speech_ms = 0.0
                self.silence_ms = 0.0
                return "stop"
        return None


class SileroCallerTracker:
    """Speech start/stop for one call, fed with its inbound PCM16 frames."""

    def __init__(
        self,
        model: SileroVadModel,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        stop_threshold: Optional[float] = None,
        start_ms: float = DEFAULT_START_MS,
        stop_ms: float = DEFAULT_STOP_MS,
    ) -> None:
        self._model = model
        self._stream: Optional[SileroVadStream] = None
        self._segmenter = SpeechSegmenter(
            threshold=threshold,
            stop_threshold=stop_threshold,
            start_ms=start_ms,
            stop_ms=stop_ms,
        )
        self.chunks = 0
        self.last_probability = 0.0
        # Monotonic times of the last chunk scored as speech and of the start
        # of the current segment, for callers that compare them with other
        # clocks (a recognizer result, a playback start).
        self.last_speech_at: Optional[float] = None
        self.segment_started_at: Optional[float] = None

    @property
    def talking(self) -> bool:
        return self._segmenter.talking

    @property
    def segment_ms(self) -> float:
        return self._segmenter.segment_ms

    @property
    def sample_rate(self) -> Optional[int]:
        return self._stream.sample_rate if self._stream is not None else None

    def feed(self, pcm16: bytes, sample_rate: int) -> List[str]:
        """Return the ``start``/``stop`` transitions this audio caused, in order."""
        if sample_rate not in CHUNK_SAMPLES:
            raise ValueError(
                f"Silero VAD runs at {SUPPORTED_SAMPLE_RATES} Hz, not {sample_rate}"
            )
        if self._stream is None or self._stream.sample_rate != sample_rate:
            self._stream = SileroVadStream(self._model, sample_rate)
            self._segmenter.reset()
        events: List[str] = []
        now = time.monotonic()
        for probability in self._stream.feed(pcm16):
            self.chunks += 1
            self.last_probability = probability
            if probability >= self._segmenter.threshold:
                self.last_speech_at = now
            event = self._segmenter.update(probability)
            if event:
                if event == "start":
                    self.segment_started_at = now
                events.append(event)
        return events


def sha256_of_file(path: str) -> str:
    return model_fetch.sha256_of_file(path)


def download_model(
    path: str,
    *,
    url: Optional[str] = None,
    sha256: Optional[str] = None,
    timeout: float = 60.0,
    opener: Optional[Callable[..., Any]] = None,
) -> str:
    """Fetch the pinned model into ``path`` atomically, verifying its SHA-256."""
    return model_fetch.download_file(
        path,
        url=url or SILERO_VAD_MODEL_URL,
        sha256=sha256 or SILERO_VAD_MODEL_SHA256,
        timeout=timeout,
        opener=opener,
        error=SileroVadError,
    )


def ensure_model_file(
    path: str = DEFAULT_MODEL_PATH,
    *,
    auto_download: bool = True,
    timeout: float = 60.0,
    opener: Optional[Callable[..., Any]] = None,
) -> str:
    """Return ``path`` once a usable model file is there, fetching it if allowed.

    A present file is loaded even when it is not the pinned release, so a
    deployment can drop in a different Silero build on purpose; the mismatch
    is logged because a truncated copy would otherwise be hard to tell apart.
    """
    return model_fetch.ensure_file(
        path,
        url=SILERO_VAD_MODEL_URL,
        sha256=SILERO_VAD_MODEL_SHA256,
        version=SILERO_VAD_VERSION,
        auto_download=auto_download,
        timeout=timeout,
        opener=opener,
        fetch_hint=f"run {FETCH_SCRIPT} or set vad.silero_auto_download: true",
        error=SileroVadError,
        label="Silero VAD model",
    )


_MODEL_CACHE: Dict[str, SileroVadModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model(path: str) -> SileroVadModel:
    """Load (once per path) the ONNX session with a single CPU thread."""
    if not ONNXRUNTIME_AVAILABLE:
        raise SileroVadError(
            "onnxruntime is not installed; add it to the engine image "
            "(requirements.txt) to use Silero VAD"
        )
    key = os.path.abspath(path)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        options = _onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        try:
            session = _onnxruntime.InferenceSession(
                key, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise SileroVadError(f"cannot load the Silero VAD model at {path}: {exc}") from exc
        model = SileroVadModel(session, key)
        _MODEL_CACHE[key] = model
        return model


def describe() -> Dict[str, Any]:
    """Static facts for startup logs and the fetch script."""
    return {
        "version": SILERO_VAD_VERSION,
        "url": SILERO_VAD_MODEL_URL,
        "sha256": SILERO_VAD_MODEL_SHA256,
        "bytes": SILERO_VAD_MODEL_BYTES,
        "chunk_ms": CHUNK_MS,
        "sample_rates": list(SUPPORTED_SAMPLE_RATES),
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
    }
