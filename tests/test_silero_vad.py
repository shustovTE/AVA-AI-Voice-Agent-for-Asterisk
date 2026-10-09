"""Silero VAD wrapper: chunking, hysteresis, and model acquisition.

The ONNX graph itself is exercised only when onnxruntime and the pinned model
file are present (see ``test_real_model``); everything else runs against a
scripted stand-in so the segmenter's timing is checked deterministically.
"""

import hashlib
import io
import os

import numpy as np
import pytest

from src.core import silero_vad as sv


class _ScriptedModel:
    """Returns scripted probabilities, records every window it was given."""

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.windows = []
        self.rates = []

    def run(self, samples, state, sample_rate):
        self.windows.append(np.array(samples, copy=True))
        self.rates.append(sample_rate)
        probability = self.probabilities.pop(0) if self.probabilities else 0.0
        return probability, state


def _pcm(sample_count, value=1000):
    return np.full(sample_count, value, dtype="<i2").tobytes()


# --- stream ------------------------------------------------------------------


def test_stream_scores_one_probability_per_full_chunk_and_keeps_the_remainder():
    model = _ScriptedModel([0.1, 0.9, 0.2])
    stream = sv.SileroVadStream(model, 16000)

    # 20 ms AudioSocket frames at 16 kHz are 320 samples: no chunk yet.
    assert stream.feed(_pcm(320)) == []
    # 640 samples buffered: one 512-sample chunk, 128 left over.
    assert stream.feed(_pcm(320)) == [0.1]
    assert len(stream._pending) == 128
    # Two more frames complete the second chunk (768 - 512 = 256 pending).
    assert stream.feed(_pcm(640)) == [0.9]
    assert len(stream._pending) == 256
    assert model.rates == [16000, 16000]


def test_stream_window_carries_the_previous_chunk_context():
    model = _ScriptedModel([0.0, 0.0])
    stream = sv.SileroVadStream(model, 8000)

    first = np.arange(256, dtype="<i2").tobytes()
    second = np.arange(256, 512, dtype="<i2").tobytes()
    stream.feed(first)
    stream.feed(second)

    assert model.windows[0].shape == (1, 32 + 256)
    # The first window starts with a zero context.
    assert np.all(model.windows[0][0, :32] == 0)
    # The second starts with the last 32 samples of the first chunk.
    expected = np.arange(224, 256, dtype=np.float32) / 32768.0
    assert np.allclose(model.windows[1][0, :32], expected)


def test_stream_rejects_unsupported_rates_and_odd_bytes():
    model = _ScriptedModel([0.0])
    with pytest.raises(ValueError):
        sv.SileroVadStream(model, 44100)
    stream = sv.SileroVadStream(model, 16000)
    # An odd trailing byte is dropped rather than raising.
    assert stream.feed(b"\x00" * 1025) == [0.0]


# --- segmenter -----------------------------------------------------------------


def test_speech_starts_after_start_ms_of_consecutive_speech():
    seg = sv.SpeechSegmenter(threshold=0.5, start_ms=96, stop_ms=300)

    assert seg.update(0.9) is None
    assert seg.update(0.9) is None
    assert seg.update(0.9) == "start"
    assert seg.talking


def test_a_blip_shorter_than_start_ms_is_not_speech():
    seg = sv.SpeechSegmenter(threshold=0.5, start_ms=96, stop_ms=300)

    assert seg.update(0.9) is None
    assert seg.update(0.9) is None
    assert seg.update(0.1) is None  # below the stop threshold: candidate reset
    assert seg.update(0.9) is None
    assert not seg.talking


def test_speech_stops_after_stop_ms_of_silence():
    seg = sv.SpeechSegmenter(threshold=0.5, start_ms=0, stop_ms=96)

    assert seg.update(0.9) == "start"
    assert seg.update(0.1) is None  # 32 ms
    assert seg.update(0.1) is None  # 64 ms
    assert seg.update(0.1) == "stop"  # 96 ms
    assert not seg.talking


def test_speech_within_the_segment_resets_the_silence():
    seg = sv.SpeechSegmenter(threshold=0.5, start_ms=0, stop_ms=96)

    seg.update(0.9)
    seg.update(0.1)
    seg.update(0.1)
    assert seg.update(0.8) is None  # speech again: silence restarts
    assert seg.update(0.1) is None
    assert seg.update(0.1) is None
    assert seg.update(0.1) == "stop"


def test_probabilities_between_the_thresholds_neither_end_nor_restart_speech():
    """Silero's own hysteresis: the band between stop and start is neutral."""
    seg = sv.SpeechSegmenter(threshold=0.5, start_ms=0, stop_ms=64)

    seg.update(0.9)
    assert seg.update(0.45) is None  # in the band: no silence started
    assert seg.update(0.45) is None
    assert seg.talking
    assert seg.update(0.2) is None  # silence starts here
    assert seg.update(0.45) == "stop"  # the band does not reset a started silence


def test_stop_threshold_defaults_to_silero_margin_and_never_exceeds_threshold():
    assert sv.SpeechSegmenter(threshold=0.5).stop_threshold == pytest.approx(0.35)
    assert sv.SpeechSegmenter(threshold=0.1).stop_threshold == pytest.approx(0.0)
    assert sv.SpeechSegmenter(threshold=0.5, stop_threshold=0.9).stop_threshold == 0.5


# --- tracker -------------------------------------------------------------------


def test_tracker_reports_transitions_and_speech_timestamps():
    model = _ScriptedModel([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    tracker = sv.SileroCallerTracker(model, threshold=0.5, start_ms=96, stop_ms=300)
    chunk = _pcm(512)

    assert tracker.feed(chunk, 16000) == []
    assert tracker.last_speech_at is not None
    assert tracker.feed(chunk, 16000) == []
    assert tracker.feed(chunk, 16000) == ["start"]
    assert tracker.talking
    assert tracker.segment_started_at is not None
    last_speech = tracker.last_speech_at
    events = []
    for _ in range(10):
        events.extend(tracker.feed(chunk, 16000))
    assert events == ["stop"]
    assert not tracker.talking
    # Silence never moved the last-speech mark.
    assert tracker.last_speech_at == last_speech
    assert tracker.chunks == 13


def test_tracker_resets_when_the_wire_rate_changes():
    model = _ScriptedModel([0.9] * 10)
    tracker = sv.SileroCallerTracker(model, start_ms=0)

    assert tracker.feed(_pcm(512), 16000) == ["start"]
    assert tracker.feed(_pcm(256), 8000) == ["start"]
    assert tracker.sample_rate == 8000


def test_tracker_rejects_unsupported_rates():
    tracker = sv.SileroCallerTracker(_ScriptedModel([]))
    with pytest.raises(ValueError):
        tracker.feed(_pcm(480), 48000)


# --- model file ----------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _opener_returning(payload):
    def opener(url, timeout):
        return _Response(payload)

    return opener


def test_missing_model_without_auto_download_names_the_fix(tmp_path):
    path = tmp_path / "vad" / "silero_vad.onnx"
    with pytest.raises(sv.SileroVadError) as excinfo:
        sv.ensure_model_file(str(path), auto_download=False)
    assert sv.FETCH_SCRIPT in str(excinfo.value)
    assert not path.exists()


def test_download_verifies_the_pinned_checksum(tmp_path, monkeypatch):
    path = tmp_path / "vad" / "silero_vad.onnx"
    with pytest.raises(sv.SileroVadError) as excinfo:
        sv.ensure_model_file(str(path), auto_download=True, opener=_opener_returning(b"not a model"))
    assert "sha256" in str(excinfo.value)
    # Nothing half-written is left behind.
    assert not path.exists()
    assert list((tmp_path / "vad").iterdir()) == []


def test_download_lands_atomically_when_the_checksum_matches(tmp_path, monkeypatch):
    payload = b"pretend model bytes"
    monkeypatch.setattr(sv, "SILERO_VAD_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    path = tmp_path / "vad" / "silero_vad.onnx"

    assert sv.ensure_model_file(str(path), auto_download=True, opener=_opener_returning(payload)) == str(path)
    assert path.read_bytes() == payload
    # A second call finds the file and does not fetch again.
    assert sv.ensure_model_file(str(path), auto_download=False) == str(path)


def test_a_present_but_different_file_is_kept_with_a_warning(tmp_path):
    path = tmp_path / "silero_vad.onnx"
    path.write_bytes(b"some other build")

    assert sv.ensure_model_file(str(path), auto_download=False) == str(path)


def test_load_model_without_onnxruntime_explains_the_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "ONNXRUNTIME_AVAILABLE", False)
    with pytest.raises(sv.SileroVadError) as excinfo:
        sv.load_model(str(tmp_path / "silero_vad.onnx"))
    assert "onnxruntime" in str(excinfo.value)


def test_describe_reports_the_pinned_release():
    facts = sv.describe()
    assert facts["version"] == sv.SILERO_VAD_VERSION
    assert facts["sha256"] == sv.SILERO_VAD_MODEL_SHA256
    assert facts["sample_rates"] == [8000, 16000]


# --- the real graph, when available ---------------------------------------------

_REAL_MODEL = os.environ.get("SILERO_VAD_MODEL_PATH") or sv.DEFAULT_MODEL_PATH


@pytest.mark.skipif(
    not sv.ONNXRUNTIME_AVAILABLE or not os.path.isfile(_REAL_MODEL),
    reason="onnxruntime and the Silero VAD model file are needed",
)
def test_real_model_scores_silence_and_noise_as_not_speech():
    model = sv.load_model(_REAL_MODEL)
    rng = np.random.default_rng(0)
    for rate in (8000, 16000):
        stream = sv.SileroVadStream(model, rate)
        silence = stream.feed(b"\x00" * rate * 2)
        noise = stream.feed((rng.standard_normal(rate) * 3000).astype("<i2").tobytes())
        assert max(silence) < 0.1
        assert max(noise) < 0.3
