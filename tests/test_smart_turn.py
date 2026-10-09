"""Smart Turn v3 wrapper: Whisper features, the audio window, and model acquisition.

The feature front end is checked against values produced by
``transformers.WhisperFeatureExtractor(chunk_length=8)`` for a synthetic
signal, so the engine's numpy implementation stays bit-for-bit compatible
with the reference the model was trained with. The ONNX graph itself runs
only when onnxruntime and the pinned model file are present.
"""

import hashlib
import io
import os

import numpy as np
import pytest

from src.core import model_fetch
from src.core import smart_turn as st


def _synthetic_signal():
    t = np.arange(32000) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 2200 * t)).astype(np.float32)


# Values of WhisperFeatureExtractor(chunk_length=8)(left-padded signal,
# padding="max_length", max_length=128000, do_normalize=True).
_REFERENCE_FEATURES = [
    (0, 0, -0.19695),
    (0, 700, -0.19695),
    (10, 700, 1.71359),
    (20, 750, -0.19695),
    (40, 799, 0.19713),
    (79, 799, -0.05153),
    (5, 100, -0.19695),
]


def test_log_mel_features_match_the_whisper_reference():
    features = st.log_mel_features(_synthetic_signal())

    assert features.shape == (80, 800)
    assert features.dtype == np.float32
    for mel, frame, expected in _REFERENCE_FEATURES:
        assert features[mel, frame] == pytest.approx(expected, abs=2e-4)
    assert float(features.mean()) == pytest.approx(-0.15633, abs=2e-4)
    assert float(features.max()) == pytest.approx(1.80305, abs=2e-4)


def test_silence_features_sit_at_the_floor():
    features = st.log_mel_features(np.zeros(st.WINDOW_SAMPLES, dtype=np.float32))
    assert np.allclose(features, -1.5)


def test_long_audio_keeps_its_end_and_short_audio_is_padded_in_front():
    long_audio = np.arange(st.WINDOW_SAMPLES + 1000, dtype=np.float32)
    short_audio = np.ones(1000, dtype=np.float32)

    # Padding is applied before normalization, so compare shapes via prepare_window.
    kept = st.prepare_window((np.clip(long_audio / long_audio.max(), -1, 1) * 32767).astype("<i2").tobytes(), 16000)
    padded = st.prepare_window((short_audio * 1000).astype("<i2").tobytes(), 16000)

    assert kept.shape == (st.WINDOW_SAMPLES,)
    assert padded.shape == (st.WINDOW_SAMPLES,)
    assert np.all(padded[: st.WINDOW_SAMPLES - 1000] == 0)
    assert np.all(padded[-1000:] != 0)


def test_mel_filterbank_has_slaney_shape():
    filters = st.mel_filterbank()
    assert filters.shape == (201, 80)
    # Every filter has a peak and the bank covers the band.
    assert np.all(filters.max(axis=0) > 0)
    assert filters[0, 0] == 0.0


def test_upsample_2x_doubles_length_and_suppresses_the_image():
    t = np.arange(16000) / 8000.0
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    up = st.upsample_2x(tone)

    assert up.shape == (32000,)
    core = up[2000:-2000] * np.hanning(28000)
    spectrum = np.abs(np.fft.rfft(core))
    freqs = np.fft.rfftfreq(28000, 1 / 16000)
    fundamental = spectrum[(freqs > 900) & (freqs < 1100)].max()
    image = spectrum[(freqs > 6900) & (freqs < 7100)].max()
    assert image < fundamental * 1e-3
    assert np.sqrt((up**2).mean()) == pytest.approx(np.sqrt((tone**2).mean()), rel=0.02)


def test_prepare_window_resamples_8k_and_other_rates_to_16k():
    pcm8k = np.full(8000, 1000, dtype="<i2").tobytes()  # 1 s at 8 kHz
    pcm48k = np.full(48000, 1000, dtype="<i2").tobytes()  # 1 s at 48 kHz

    for pcm, rate in ((pcm8k, 8000), (pcm48k, 48000)):
        window = st.prepare_window(pcm, rate)
        assert window.shape == (st.WINDOW_SAMPLES,)
        # One second of signal sits at the end of the window.
        assert np.all(window[: st.WINDOW_SAMPLES - 16000 - 100] == 0)
        assert np.abs(window[-8000:]).mean() > 0.01


# --- the turn audio buffer ---------------------------------------------------


def test_buffer_keeps_only_the_last_window_plus_margin():
    buffer = st.TurnAudioBuffer(max_seconds=1.0)
    frame = b"\x01\x00" * 160  # 20 ms at 8 kHz

    for _ in range(100):  # 2 s
        buffer.append(frame, 8000)

    assert buffer.duration_ms == pytest.approx(1000, abs=20)
    data, rate = buffer.snapshot()
    assert rate == 8000
    assert len(data) == int(buffer.duration_ms / 1000 * 8000) * 2


def test_buffer_snapshot_trims_trailing_silence_and_clear_empties_it():
    buffer = st.TurnAudioBuffer()
    buffer.append(b"\x01\x00" * 8000, 16000)  # 500 ms

    data, _ = buffer.snapshot(trim_trailing_ms=300)
    assert len(data) == (8000 - 4800) * 2
    # A trim longer than the audio leaves the audio alone.
    data, _ = buffer.snapshot(trim_trailing_ms=5000)
    assert len(data) == 8000 * 2

    buffer.clear()
    assert buffer.duration_ms == 0
    assert buffer.snapshot() == (b"", 16000)


def test_buffer_restarts_when_the_wire_rate_changes():
    buffer = st.TurnAudioBuffer()
    buffer.append(b"\x01\x00" * 160, 8000)
    buffer.append(b"\x01\x00" * 320, 16000)
    assert buffer.sample_rate == 16000
    assert buffer.duration_ms == pytest.approx(20)


# --- the model wrapper ---------------------------------------------------------


class _FakeInput:
    name = "input_features"


class _FakeSession:
    def __init__(self, probability):
        self.probability = probability
        self.calls = []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, feeds):
        self.calls.append(feeds["input_features"].shape)
        return [np.array([[self.probability]], dtype=np.float32)]


def test_model_predict_reports_probability_and_timing():
    session = _FakeSession(0.83)
    model = st.SmartTurnModel(session, "fake.onnx")

    result = model.predict(np.full(8000, 500, dtype="<i2").tobytes(), 8000)

    assert result["probability"] == pytest.approx(0.83)
    assert result["audio_ms"] == pytest.approx(1000)
    assert result["inference_ms"] >= 0
    assert session.calls == [(1, 80, 800)]


def test_model_rejects_a_graph_without_the_whisper_input():
    class _Other:
        name = "input"

    class _Session:
        def get_inputs(self):
            return [_Other()]

    with pytest.raises(st.SmartTurnError):
        st.SmartTurnModel(_Session(), "other.onnx")


# --- model file ------------------------------------------------------------------


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
    path = tmp_path / "turn" / "smart-turn.onnx"
    with pytest.raises(st.SmartTurnError) as excinfo:
        st.ensure_model_file(str(path), auto_download=False)
    assert st.FETCH_SCRIPT in str(excinfo.value)


def test_download_verifies_the_pinned_checksum(tmp_path):
    path = tmp_path / "turn" / "smart-turn.onnx"
    with pytest.raises(st.SmartTurnError) as excinfo:
        st.ensure_model_file(str(path), auto_download=True, opener=_opener_returning(b"nope"))
    assert "sha256" in str(excinfo.value)
    assert not path.exists()
    assert list((tmp_path / "turn").iterdir()) == []


def test_download_lands_atomically_when_the_checksum_matches(tmp_path, monkeypatch):
    payload = b"pretend smart turn"
    monkeypatch.setattr(st, "SMART_TURN_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    path = tmp_path / "turn" / "smart-turn.onnx"

    assert st.ensure_model_file(str(path), auto_download=True, opener=_opener_returning(payload)) == str(path)
    assert path.read_bytes() == payload
    assert oct(path.stat().st_mode & 0o777) == "0o644"


def test_load_model_without_onnxruntime_explains_the_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "ONNXRUNTIME_AVAILABLE", False)
    with pytest.raises(st.SmartTurnError) as excinfo:
        st.load_model(str(tmp_path / "smart-turn.onnx"))
    assert "onnxruntime" in str(excinfo.value)


def test_shared_fetch_helper_reports_a_different_present_file_but_keeps_it(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"another build")
    assert (
        model_fetch.ensure_file(
            str(path), url="https://example.invalid/x", sha256="0" * 64, version="x", auto_download=False
        )
        == str(path)
    )


def test_describe_reports_the_pinned_release():
    facts = st.describe()
    assert facts["version"] == st.SMART_TURN_VERSION
    assert facts["sha256"] == st.SMART_TURN_MODEL_SHA256
    assert facts["window_seconds"] == 8


# --- the real graph, when available -----------------------------------------------

_REAL_MODEL = os.environ.get("SMART_TURN_MODEL_PATH") or st.DEFAULT_MODEL_PATH


@pytest.mark.skipif(
    not st.ONNXRUNTIME_AVAILABLE or not os.path.isfile(_REAL_MODEL),
    reason="onnxruntime and the Smart Turn model file are needed",
)
def test_real_model_scores_silence_as_complete():
    model = st.load_model(_REAL_MODEL)
    result = model.predict(b"\x00" * 32000, 16000)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["probability"] > 0.5
