"""GigaAM v3 and NeMo FastConformer RU in the local AI server through onnx-asr.

The only Russian recognizer the local AI server had was the streaming T-one;
the offline Russian models (GigaAM v3, NeMo FastConformer RU) decode whole
phrases and had no way in. ``LOCAL_STT_BACKEND=onnx_asr`` runs them behind
the Silero VAD gate the Sherpa offline backend already uses: a phrase is
recognized once the caller pauses, on CUDA when the GPU image and a GPU are
present. The model is fetched from Hugging Face by name on first start.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

LOCAL_AI_DIR = str(Path(__file__).resolve().parents[1] / "local_ai_server")
ADMIN_UI_DIR = str(Path(__file__).resolve().parents[1] / "admin_ui" / "backend")

try:
    import fastapi  # noqa: F401

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def _load(name: str, directory: str = LOCAL_AI_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module(name)


# --- fakes for the two optional packages --------------------------------------------


class _FakeRecognizer:
    def __init__(self, text="привет мир"):
        self.text = text
        self.calls = []

    def recognize(self, waveform, *, sample_rate=16_000):
        self.calls.append((waveform, sample_rate))
        return self.text


class _FakeManager:
    """Stands in for onnx_asr.loader.Manager and records how the model was created."""

    instances = []

    def __init__(self, providers=None, **kwargs):
        self.providers = providers
        self.create_calls = []
        _FakeManager.instances.append(self)

    def create_asr(self, model, local_dir, *, quantization=None, offline=None, config=None):
        self.create_calls.append({"model": model, "local_dir": local_dir, "quantization": quantization, "offline": offline})
        return _FakeRecognizer()


def _fake_onnx_asr():
    module = ModuleType("onnx_asr")
    loader = ModuleType("onnx_asr.loader")
    loader.Manager = _FakeManager
    module.loader = loader
    return module


def _fake_sherpa():
    module = ModuleType("sherpa_onnx")

    class VadModelConfig:
        def __init__(self):
            self.silero_vad = SimpleNamespace()
            self.sample_rate = None

    module.VadModelConfig = VadModelConfig
    module.VoiceActivityDetector = MagicMock()
    return module


def _fake_onnxruntime(providers):
    module = ModuleType("onnxruntime")
    module.get_available_providers = lambda: list(providers)
    return module


class _FakeSpeechSegment:
    def __init__(self, samples, start=0):
        self.samples = samples
        self.start = start  # sherpa-onnx: the segment's first sample, counted from the VAD's first sample


class _FakeVAD:
    def __init__(self, segments=None):
        self._segments = list(segments or [])
        self.flushed = False

    def accept_waveform(self, samples):
        pass

    def empty(self):
        return not self._segments

    @property
    def front(self):
        return self._segments[0]

    def pop(self):
        self._segments.pop(0)

    def flush(self):
        self.flushed = True


def _backend(**overrides):
    sb = _load("stt_backends")
    kwargs = dict(model="gigaam-v3-e2e-ctc", vad_model_path="/fake/vad.onnx", cache_dir="/models/onnx-asr")
    kwargs.update(overrides)
    return sb.OnnxAsrSTTBackend(**kwargs)


# --- configuration ------------------------------------------------------------------


def _config(monkeypatch, **env):
    for key in ("LOCAL_STT_BACKEND", "ONNX_ASR_MODEL", "ONNX_ASR_MODEL_PATH", "ONNX_ASR_CACHE_DIR", "ONNX_ASR_QUANTIZATION", "ONNX_ASR_DEVICE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return _load("config").LocalAIConfig.from_env()


def test_config_defaults_to_gigaam_v3_e2e_ctc_downloaded_into_the_models_volume(monkeypatch):
    cfg = _config(monkeypatch)
    assert cfg.onnx_asr_model == "gigaam-v3-e2e-ctc"
    assert cfg.onnx_asr_model_path == ""
    assert cfg.onnx_asr_cache_dir == "/app/models/stt/onnx-asr"
    assert cfg.onnx_asr_quantization == ""
    assert cfg.onnx_asr_device == "auto"


def test_config_reads_the_onnx_asr_environment(monkeypatch):
    cfg = _config(
        monkeypatch,
        LOCAL_STT_BACKEND="onnx_asr",
        ONNX_ASR_MODEL=" gigaam-v3-rnnt ",
        ONNX_ASR_MODEL_PATH="/app/models/stt/my-gigaam",
        ONNX_ASR_CACHE_DIR="/data/asr",
        ONNX_ASR_QUANTIZATION="INT8",
        ONNX_ASR_DEVICE=" CUDA ",
    )
    assert cfg.stt_backend == "onnx_asr"
    assert cfg.onnx_asr_model == "gigaam-v3-rnnt"
    assert cfg.onnx_asr_model_path == "/app/models/stt/my-gigaam"
    assert cfg.onnx_asr_cache_dir == "/data/asr"
    assert cfg.onnx_asr_quantization == "int8"
    assert cfg.onnx_asr_device == "cuda"



def test_config_takes_a_directory_in_onnx_asr_model_apart(monkeypatch, tmp_path):
    """ONNX_ASR_MODEL=/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc is a directory, which onnx-asr would read as a repo id."""
    # A missing or empty directory: the name is kept, nothing goes offline.
    cfg = _config(monkeypatch, ONNX_ASR_MODEL=str(tmp_path / "onnx-asr" / "gigaam-v3-e2e-ctc"))
    assert cfg.onnx_asr_model == "gigaam-v3-e2e-ctc" and cfg.onnx_asr_model_path == ""

    # A directory with files becomes the model directory of the model named after it.
    model_dir = tmp_path / "onnx-asr" / "t-tech__t-one"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    cfg = _config(monkeypatch, ONNX_ASR_MODEL=str(model_dir))
    assert cfg.onnx_asr_model == "t-tech/t-one" and cfg.onnx_asr_model_path == str(model_dir)

    # An explicit ONNX_ASR_MODEL_PATH wins over the directory in the name; a repo id is a name.
    cfg = _config(monkeypatch, ONNX_ASR_MODEL=str(model_dir), ONNX_ASR_MODEL_PATH="/data/custom")
    assert cfg.onnx_asr_model == "t-tech/t-one" and cfg.onnx_asr_model_path == "/data/custom"
    cfg = _config(monkeypatch, ONNX_ASR_MODEL="alphacep/vosk-model-ru")
    assert cfg.onnx_asr_model == "alphacep/vosk-model-ru" and cfg.onnx_asr_model_path == ""


# --- the backend --------------------------------------------------------------------


def test_the_backend_is_the_offline_vad_gate_with_an_onnx_asr_model():
    sb = _load("stt_backends")
    backend = _backend(
        preroll_ms=350,
        vad_threshold=0.35,
        vad_min_silence_ms=700,
        vad_min_speech_ms=200,
        postroll_ms=300,
        normalize_dbfs=-20,
        normalize_max_gain_db=18,
    )

    assert isinstance(backend, sb.SherpaOfflineSTTBackend)
    assert backend.LOG_TAG == "ONNX-ASR"
    assert backend.model_dir == "/models/onnx-asr/gigaam-v3-e2e-ctc"
    assert backend.model_path == backend.model_dir
    assert backend.quantization is None
    assert backend.device == "auto"
    assert (backend.preroll_ms, backend.vad_threshold, backend.vad_min_silence_ms, backend.vad_min_speech_ms) == (350, 0.35, 700, 200)
    assert (backend.postroll_ms, backend.normalize_dbfs, backend.normalize_max_gain_db) == (300, -20.0, 18.0)
    # The decode floor follows the VAD's own minimum, so a phrase the VAD accepted is never dropped.
    assert backend._min_audio_length == 3200
    assert backend.tuning_summary() == (
        "(preroll_ms=350 postroll_ms=300 threshold=0.35 min_silence_ms=700 min_speech_ms=200 "
        "normalize_dbfs=-20.0 max_gain_db=18)"
    )
    # The Sherpa offline backend keeps its own tag.
    assert sb.SherpaOfflineSTTBackend.LOG_TAG == "SHERPA-OFFLINE"


def test_model_names_paths_quantization_and_devices_are_normalized():
    assert _backend(model="alphacep/vosk-model-ru").model_dir == "/models/onnx-asr/alphacep__vosk-model-ru"
    assert _backend(model_path="/data/gigaam").model_dir == "/data/gigaam"
    assert _backend(model="").model == "gigaam-v3-e2e-ctc"
    assert _backend(quantization="fp32").quantization is None
    assert _backend(quantization="INT8").quantization == "int8"
    assert _backend(device="cuda").device == "cuda"
    assert _backend(device="gpu").device == "auto"  # unknown values fall back to auto


@pytest.mark.parametrize(
    "device, available, expected",
    [
        ("auto", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("auto", ["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        ("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
        ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("cuda", ["CPUExecutionProvider"], ["CPUExecutionProvider"]),  # no GPU: warns and runs on the CPU
    ],
)
def test_execution_providers_follow_the_device_and_what_onnxruntime_offers(device, available, expected):
    backend = _backend(device=device)
    with patch.dict(sys.modules, {"onnxruntime": _fake_onnxruntime(available)}):
        assert backend._select_providers() == expected


def test_initialize_downloads_by_name_into_the_cache_and_builds_the_vad(tmp_path):
    vad = tmp_path / "silero_vad.onnx"
    vad.write_bytes(b"vad")
    backend = _backend(vad_model_path=str(vad), cache_dir=str(tmp_path / "onnx-asr"), device="cpu")
    _FakeManager.instances.clear()

    with patch.dict(
        sys.modules,
        {"onnx_asr": _fake_onnx_asr(), "sherpa_onnx": _fake_sherpa(), "onnxruntime": _fake_onnxruntime(["CPUExecutionProvider"])},
    ):
        assert backend.initialize() is True

    manager = _FakeManager.instances[-1]
    assert manager.providers == ["CPUExecutionProvider"]
    assert manager.create_calls == [
        {"model": "gigaam-v3-e2e-ctc", "local_dir": str(tmp_path / "onnx-asr" / "gigaam-v3-e2e-ctc"), "quantization": None, "offline": False}
    ]
    assert (tmp_path / "onnx-asr" / "gigaam-v3-e2e-ctc").is_dir()
    assert backend._initialized is True
    assert backend._vad_config is not None
    assert backend._vad_config.silero_vad.model == str(vad)
    assert backend.providers == ["CPUExecutionProvider"]


def test_initialize_uses_an_explicit_model_directory_offline(tmp_path):
    vad = tmp_path / "silero_vad.onnx"
    vad.write_bytes(b"vad")
    backend = _backend(vad_model_path=str(vad), model_path="/data/gigaam", quantization="int8", device="cuda")
    _FakeManager.instances.clear()

    with patch.dict(
        sys.modules,
        {"onnx_asr": _fake_onnx_asr(), "sherpa_onnx": _fake_sherpa(), "onnxruntime": _fake_onnxruntime(["CUDAExecutionProvider", "CPUExecutionProvider"])},
    ):
        assert backend.initialize() is True

    manager = _FakeManager.instances[-1]
    assert manager.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert manager.create_calls == [{"model": "gigaam-v3-e2e-ctc", "local_dir": "/data/gigaam", "quantization": "int8", "offline": True}]


def test_initialize_explains_a_missing_package_or_vad(tmp_path, caplog):
    vad = tmp_path / "silero_vad.onnx"
    vad.write_bytes(b"vad")

    with patch.dict(sys.modules, {"onnx_asr": None, "sherpa_onnx": _fake_sherpa()}):
        assert _backend(vad_model_path=str(vad)).initialize() is False
    assert "INCLUDE_ONNX_ASR=true" in caplog.text

    with patch.dict(sys.modules, {"onnx_asr": _fake_onnx_asr(), "sherpa_onnx": None}):
        assert _backend(vad_model_path=str(vad)).initialize() is False
    assert "INCLUDE_SHERPA=true" in caplog.text

    with patch.dict(sys.modules, {"onnx_asr": _fake_onnx_asr(), "sherpa_onnx": _fake_sherpa()}):
        assert _backend(vad_model_path=str(tmp_path / "missing.onnx")).initialize() is False
    assert "Silero VAD model not found" in caplog.text


def test_initialize_explains_an_unwritable_models_volume(tmp_path, caplog, monkeypatch):
    """A root-owned ./models bind mount: the error names the uid and the host command instead of a bare errno."""
    vad = tmp_path / "silero_vad.onnx"
    vad.write_bytes(b"vad")
    backend = _backend(vad_model_path=str(vad), cache_dir="/app/models/stt/onnx-asr", device="cpu")
    _FakeManager.instances.clear()

    def denied(path, exist_ok=False):
        raise PermissionError(13, "Permission denied", "/app/models/stt")

    monkeypatch.setattr(_load("stt_backends").os, "makedirs", denied)
    with patch.dict(
        sys.modules,
        {"onnx_asr": _fake_onnx_asr(), "sherpa_onnx": _fake_sherpa(), "onnxruntime": _fake_onnxruntime(["CPUExecutionProvider"])},
    ):
        assert backend.initialize() is False

    assert _FakeManager.instances == []  # nothing was downloaded or loaded
    assert "not writable by the container user" in caplog.text
    assert "sudo chown -R" in caplog.text
    assert "/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc" in caplog.text
    assert backend._initialized is False


def test_a_vad_segment_is_decoded_by_the_onnx_asr_model():
    backend = _backend()
    backend.recognizer = _FakeRecognizer("  Привет, мир!  ")
    backend._vad_config = "fake"
    backend._initialized = True
    samples = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)  # one second of "speech"

    result = backend.process_audio(_FakeVAD([_FakeSpeechSegment(samples.tolist())]), b"\x00\x00" * 320)

    assert result == {"type": "final", "text": "Привет, мир!"}
    waveform, sample_rate = backend.recognizer.calls[0]
    assert sample_rate == 16_000
    assert waveform.dtype == np.float32 and waveform.flags["C_CONTIGUOUS"]
    assert len(waveform) == 16_000


def test_finalize_flushes_the_vad_and_decodes_the_trailing_speech():
    backend = _backend()
    backend.recognizer = _FakeRecognizer("до свидания")
    backend._vad_config = "fake"
    backend._initialized = True
    vad = _FakeVAD([_FakeSpeechSegment(np.full(8_000, 0.1, dtype=np.float32).tolist())])

    assert backend.finalize(vad) == {"type": "final", "text": "до свидания"}
    assert vad.flushed is True
    assert backend.finalize(_FakeVAD()) is None


def _ready_backend(text="да", **overrides):
    backend = _backend(**overrides)
    backend.recognizer = _FakeRecognizer(text)
    backend._vad_config = "fake"
    backend._initialized = True
    return backend


def _pcm16(value: float, samples: int) -> bytes:
    return np.full(samples, int(value * 32768), dtype=np.int16).tobytes()


def test_a_segment_is_widened_with_the_stream_audio_around_it_before_decoding():
    """Pre-roll is what the caller said before the VAD opened, post-roll what followed the VAD's cut.

    Both come from the session's stream memory by absolute position, not from the
    tail of the stream (which, when a segment is closed, is the silence that closed it).
    """
    backend = _ready_backend(preroll_ms=100, postroll_ms=50)
    context = backend.create_session_context()
    context.bind_vad()
    # 100 ms "before" at 0.25, 500 ms of speech at 0.5, 100 ms "after" at -0.25, all through an empty VAD.
    stream = _pcm16(0.25, 1600) + _pcm16(0.5, 8000) + _pcm16(-0.25, 1600)
    assert backend.process_audio(_FakeVAD(), stream, context) is None
    assert context.total_samples == 11200

    speech = np.full(8000, 0.5, dtype=np.float32).tolist()
    result = backend.process_audio(_FakeVAD([_FakeSpeechSegment(speech, start=1600)]), b"", context)

    assert result == {"type": "final", "text": "да"}
    waveform, _ = backend.recognizer.calls[0]
    assert len(waveform) == 1600 + 8000 + 800
    assert np.allclose(waveform[:1600], 0.25, atol=1e-4)  # 100 ms of pre-roll from before the phrase
    assert np.allclose(waveform[1600:9600], 0.5, atol=1e-4)  # the VAD's own speech
    assert np.allclose(waveform[9600:], -0.25, atol=1e-4)  # 50 ms of post-roll from after it


def test_a_vad_created_after_a_final_is_mapped_onto_the_stream_where_it_started():
    """The server recreates the VAD after every final; its sample 0 sits where the stream is by then."""
    backend = _ready_backend(preroll_ms=100, postroll_ms=0)
    context = backend.create_session_context()
    context.bind_vad()
    assert backend.process_audio(_FakeVAD(), _pcm16(0.1, 16_000), context) is None  # one second, first VAD
    context.bind_vad()  # a new VAD: its sample 0 is stream sample 16000
    assert backend.process_audio(_FakeVAD(), _pcm16(0.3, 16_000), context) is None

    speech = np.full(4000, 0.5, dtype=np.float32).tolist()
    backend.process_audio(_FakeVAD([_FakeSpeechSegment(speech, start=800)]), b"", context)

    waveform, _ = backend.recognizer.calls[0]
    # The segment starts at stream sample 16800; its 1600-sample pre-roll spans both seconds.
    assert len(waveform) == 1600 + 4000
    assert np.allclose(waveform[:800], 0.1, atol=1e-4)
    assert np.allclose(waveform[800:1600], 0.3, atol=1e-4)


def test_finalize_pads_the_post_roll_of_a_flushed_segment_with_silence():
    """A flushed segment ends at the last sample fed: nothing follows it, so its post-roll is zeros."""
    backend = _ready_backend(preroll_ms=0, postroll_ms=100)
    context = backend.create_session_context()
    context.bind_vad()
    backend.process_audio(_FakeVAD(), _pcm16(0.5, 8000), context)

    speech = np.full(8000, 0.5, dtype=np.float32).tolist()
    vad = _FakeVAD([_FakeSpeechSegment(speech, start=0)])
    assert backend.finalize(vad, context) == {"type": "final", "text": "да"}

    waveform, _ = backend.recognizer.calls[0]
    assert len(waveform) == 8000 + 1600
    assert np.allclose(waveform[:8000], 0.5, atol=1e-4)
    assert np.all(waveform[8000:] == 0.0)


def test_without_a_context_a_segment_is_decoded_as_the_vad_cut_it():
    backend = _ready_backend(preroll_ms=350, postroll_ms=300)
    speech = np.full(8000, 0.5, dtype=np.float32).tolist()
    assert backend.process_audio(_FakeVAD([_FakeSpeechSegment(speech, start=0)]), b"", None) == {"type": "final", "text": "да"}
    waveform, _ = backend.recognizer.calls[0]
    assert len(waveform) == 8000


def test_quiet_speech_is_brought_to_the_loudness_target_before_decoding():
    """Telephone speech at -40 dBFS is boosted to -20 dBFS; the boost is capped; 0 turns it off."""
    backend = _ready_backend(preroll_ms=0, postroll_ms=0, normalize_dbfs=-20.0, normalize_max_gain_db=24.0)
    quiet = (0.01 * np.sqrt(2) * np.sin(2 * np.pi * 300 * np.arange(8000) / 16_000)).astype(np.float32)  # -40 dBFS RMS

    backend.process_audio(_FakeVAD([_FakeSpeechSegment(quiet.tolist(), start=0)]), b"", None)
    waveform, _ = backend.recognizer.calls[0]
    assert 20 * np.log10(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) == pytest.approx(-20.0, abs=0.2)

    faint = (quiet / 100).astype(np.float32)  # -80 dBFS: only the 24 dB cap is applied
    backend.process_audio(_FakeVAD([_FakeSpeechSegment(faint.tolist(), start=0)]), b"", None)
    waveform, _ = backend.recognizer.calls[1]
    assert 20 * np.log10(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) == pytest.approx(-56.0, abs=0.2)

    off = _ready_backend(preroll_ms=0, postroll_ms=0, normalize_dbfs=0)
    off.process_audio(_FakeVAD([_FakeSpeechSegment(quiet.tolist(), start=0)]), b"", None)
    waveform, _ = off.recognizer.calls[0]
    assert np.allclose(waveform, quiet)


def test_shutdown_releases_the_model():
    backend = _backend()
    backend.recognizer = _FakeRecognizer()
    backend._vad_config = "fake"
    backend._initialized = True
    backend.shutdown()
    assert backend.recognizer is None and backend._vad_config is None and backend._initialized is False


# --- the server's view ----------------------------------------------------------------


def test_capabilities_report_the_package(monkeypatch):
    config = _load("config")
    capabilities = importlib.reload(_load("capabilities"))
    cfg = config.LocalAIConfig()

    monkeypatch.setitem(sys.modules, "onnx_asr", ModuleType("onnx_asr"))
    assert capabilities.detect_capabilities(cfg)["onnx_asr"] is True
    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    assert capabilities.detect_capabilities(cfg)["onnx_asr"] is False


def test_the_control_plane_switches_to_onnx_asr_and_validates_its_knobs():
    config = _load("config")
    control_plane = _load("control_plane")
    cfg = config.LocalAIConfig()

    new_cfg, changed = control_plane.apply_switch_model_request(
        cfg,
        {
            "stt_backend": "onnx_asr",
            "onnx_asr_model": "gigaam-v3-rnnt",
            "onnx_asr_device": "cuda",
            "onnx_asr_quantization": "FP32",
            "onnx_asr_model_path": "",
        },
    )
    assert new_cfg.stt_backend == "onnx_asr"
    assert new_cfg.onnx_asr_model == "gigaam-v3-rnnt"
    assert new_cfg.onnx_asr_device == "cuda"
    assert new_cfg.onnx_asr_quantization == ""
    assert "stt_backend=onnx_asr" in changed and "onnx_asr_model=gigaam-v3-rnnt" in changed
    assert "onnx_asr_quantization=fp32" in changed

    # An unknown device is ignored; a blank model keeps the current one; the generic
    # model path means the model name for this backend; stt_config keys map too.
    same_cfg, changed = control_plane.apply_switch_model_request(new_cfg, {"onnx_asr_device": "tpu", "onnx_asr_model": " "})
    assert same_cfg.onnx_asr_device == "cuda" and same_cfg.onnx_asr_model == "gigaam-v3-rnnt" and changed == []
    by_path, changed = control_plane.apply_switch_model_request(new_cfg, {"stt_model_path": "nemo-fastconformer-ru-ctc"})
    assert by_path.onnx_asr_model == "nemo-fastconformer-ru-ctc"
    by_config, changed = control_plane.apply_switch_model_request(
        new_cfg, {"stt_config": {"onnx_asr_quantization": "int8", "onnx_asr_model_path": "/data/gigaam"}}
    )
    assert by_config.onnx_asr_quantization == "int8" and by_config.onnx_asr_model_path == "/data/gigaam"


def test_status_names_the_model_the_device_and_the_language():
    status_builder = _load("status_builder")
    server = SimpleNamespace(
        stt_backend="onnx_asr",
        mock_models=False,
        onnx_asr_model="gigaam-v3-e2e-ctc",
        onnx_asr_model_path="",
        onnx_asr_device="auto",
        onnx_asr_quantization="",
        onnx_asr_backend=SimpleNamespace(providers=["CUDAExecutionProvider", "CPUExecutionProvider"], model_dir="/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc"),
    )
    loaded, path, display = status_builder._stt_status(server)
    assert loaded is True
    assert path == "/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc"
    assert display == "onnx-asr (gigaam-v3-e2e-ctc, cuda)"
    assert status_builder._stt_language(server) == "ru"

    server.onnx_asr_backend = None
    loaded, path, display = status_builder._stt_status(server)
    assert loaded is False and display == "onnx-asr (gigaam-v3-e2e-ctc, auto)"
    server.onnx_asr_model = "gigaam-multilingual-ctc"
    assert status_builder._stt_language(server) == "multi"
    server.onnx_asr_model = "nemo-fastconformer-ru-rnnt"
    assert status_builder._stt_language(server) == "ru"
    server.onnx_asr_model = "nemo-parakeet-tdt-0.6b-v2"
    assert status_builder._stt_language(server) is None


def test_the_protocol_schema_declares_the_onnx_asr_switch_fields():
    contract = _load("protocol_contract")
    schema = contract.PROTOCOL_SCHEMA
    text = str(schema)
    assert "onnx_asr_model" in text and "onnx_asr_model_path" in text
    assert "onnx_asr_quantization" in text and "onnx_asr_device" in text
    assert "'enum': ['auto', 'cpu', 'cuda']" in text


def test_the_registry_advertises_the_backend():
    stt_pkg = _load("backends.stt")
    registry = _load("backends.registry").STT_REGISTRY
    assert "onnx_asr" in registry.names()
    schema = registry.get("onnx_asr").config_schema()
    assert set(schema) == {"model", "model_path", "quantization", "device"}
    assert stt_pkg is not None



def test_the_control_plane_takes_a_directory_given_as_the_model_apart(tmp_path):
    config = _load("config")
    control_plane = _load("control_plane")
    cfg = config.LocalAIConfig(stt_backend="onnx_asr")

    by_path, changed = control_plane.apply_switch_model_request(cfg, {"stt_model_path": "/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc"})
    assert by_path.onnx_asr_model == "gigaam-v3-e2e-ctc" and by_path.onnx_asr_model_path == ""
    assert "onnx_asr_model=gigaam-v3-e2e-ctc" in changed

    model_dir = tmp_path / "gigaam-v3-rnnt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    by_model, changed = control_plane.apply_switch_model_request(cfg, {"onnx_asr_model": str(model_dir)})
    assert by_model.onnx_asr_model == "gigaam-v3-rnnt" and by_model.onnx_asr_model_path == str(model_dir)


def test_the_backend_warns_about_a_directory_given_as_the_model(caplog):
    backend = _backend(model="/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc", cache_dir="/app/models/stt/onnx-asr")
    assert backend.model == "gigaam-v3-e2e-ctc"
    assert backend.explicit_model_path == ""
    assert backend.model_dir == "/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc"
    assert "is a directory, not a model name" in caplog.text


# --- the Admin UI backend --------------------------------------------------------------


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestAdminUi:
    def _api(self):
        return _load("api.local_ai", ADMIN_UI_DIR)

    def test_switch_request_maps_to_env_yaml_and_the_ws_payload(self):
        api = self._api()
        req = api.SwitchModelRequest(
            model_type="stt",
            backend="onnx_asr",
            model_path="gigaam-v3-e2e-rnnt",
            onnx_asr_device="cuda",
            onnx_asr_quantization="",
            onnx_asr_model_path="",
        )
        env, yaml_updates = api._build_local_ai_env_and_yaml_updates(req)
        assert env["LOCAL_STT_BACKEND"] == "onnx_asr"
        assert env["ONNX_ASR_MODEL"] == "gigaam-v3-e2e-rnnt"
        assert env["ONNX_ASR_DEVICE"] == "cuda"
        assert env["ONNX_ASR_QUANTIZATION"] == ""
        assert env["ONNX_ASR_MODEL_PATH"] == ""
        assert yaml_updates["stt_backend"] == "onnx_asr" and yaml_updates["onnx_asr_model"] == "gigaam-v3-e2e-rnnt"

        payload = api._build_local_ai_ws_switch_payload(req)
        assert payload["stt_backend"] == "onnx_asr"
        assert payload["onnx_asr_model"] == "gigaam-v3-e2e-rnnt"
        assert payload["onnx_asr_device"] == "cuda"
        assert payload["onnx_asr_quantization"] == ""

        # The explicit model field wins over the generic path; unset knobs are not sent.
        req = api.SwitchModelRequest(model_type="stt", backend="onnx_asr", model_path="x", onnx_asr_model="gigaam-v3-ctc")
        assert api._build_local_ai_ws_switch_payload(req) == {"type": "switch_model", "stt_backend": "onnx_asr", "onnx_asr_model": "gigaam-v3-ctc"}

        # A container directory sent where the name belongs (the cache layout, or a hand-typed path)
        # is reduced to the model name; the env never carries a path in ONNX_ASR_MODEL.
        req = api.SwitchModelRequest(model_type="stt", backend="onnx_asr", model_path="/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc")
        env, yaml_updates = api._build_local_ai_env_and_yaml_updates(req)
        assert env["ONNX_ASR_MODEL"] == "gigaam-v3-e2e-ctc" and "ONNX_ASR_MODEL_PATH" not in env
        assert yaml_updates["onnx_asr_model"] == "gigaam-v3-e2e-ctc"
        assert api._build_local_ai_ws_switch_payload(req)["onnx_asr_model"] == "gigaam-v3-e2e-ctc"
        assert api._onnx_asr_model_name("/app/models/stt/onnx-asr/t-tech__t-one/") == "t-tech/t-one"

    def test_rebuild_request_and_build_arg_maps_know_the_backend(self):
        api = self._api()
        rebuild_jobs = _load("api.rebuild_jobs", ADMIN_UI_DIR)
        assert api.RebuildRequest(include_onnx_asr=True).include_onnx_asr is True
        assert rebuild_jobs.BACKEND_BUILD_ARGS["onnx_asr"] == "INCLUDE_ONNX_ASR"
        assert rebuild_jobs._DEFAULT_INCLUDE_BASE["onnx_asr"] is False
        assert rebuild_jobs._DEFAULT_INCLUDE_GPU["onnx_asr"] is False
        assert "onnx_asr" in rebuild_jobs.BUILD_TIME_ESTIMATES

    def test_the_catalog_lists_the_russian_models_as_server_side_downloads(self):
        catalog = _load("api.models_catalog", ADMIN_UI_DIR)
        entries = [m for m in catalog.ONNX_ASR_STT_MODELS if m["backend"] == "onnx_asr"]
        assert [m["model_path"] for m in entries] == [
            "gigaam-v3-e2e-ctc",
            "gigaam-v3-e2e-rnnt",
            "gigaam-v3-ctc",
            "gigaam-v3-rnnt",
            "nemo-fastconformer-ru-ctc",
            "nemo-fastconformer-ru-rnnt",
        ]
        assert all(m["auto_download"] is True and m["download_url"] is None for m in entries)
        assert all("INCLUDE_ONNX_ASR=true" in m["note"] for m in entries)
        assert sum(1 for m in entries if m.get("recommended")) == 1


# --- the server ---------------------------------------------------------------------


class _FakeContext:
    """What the server needs from OfflineSegmentContext: to be bound to each new VAD and handed to the backend."""

    def __init__(self):
        self.bound = 0

    def bind_vad(self):
        self.bound += 1


class _FakeServerBackend:
    """What the server needs from OnnxAsrSTTBackend: a VAD and a stream memory per session, decode, finalize, shutdown."""

    def __init__(self, results=None, trailing=None):
        self.results = list(results or [])
        self.trailing = trailing
        self.process_calls = []
        self.finalize_calls = []
        self.contexts = []
        self.shutdown_calls = 0
        self.providers = ["CPUExecutionProvider"]
        self.model_dir = "/app/models/stt/onnx-asr/gigaam-v3-e2e-ctc"

    def create_session_vad(self):
        return object()

    def create_session_context(self):
        context = _FakeContext()
        self.contexts.append(context)
        return context

    def process_audio(self, vad, pcm16, context=None):
        self.process_calls.append((vad, pcm16, context))
        return self.results.pop(0) if self.results else None

    def finalize(self, vad, context=None):
        self.finalize_calls.append((vad, context))
        return self.trailing

    def shutdown(self):
        self.shutdown_calls += 1


def _server(**attrs):
    server_mod = _load("server")
    server = server_mod.LocalAIServer.__new__(server_mod.LocalAIServer)
    server.stt_backend = "onnx_asr"
    server.mock_models = False
    server.fail_fast = False
    server.startup_errors = {}
    server.config = SimpleNamespace(
        sherpa_offline_preroll_ms=350,
        sherpa_vad_threshold=0.35,
        sherpa_vad_min_silence_ms=700,
        sherpa_vad_min_speech_ms=200,
        sherpa_offline_postroll_ms=300,
        sherpa_offline_normalize_dbfs=-20.0,
        sherpa_offline_normalize_max_gain_db=24.0,
        local_stt_resampler="fir",
    )
    server.onnx_asr_backend = None
    server.sherpa_backend = None
    server.tone_backend = None
    for key, value in attrs.items():
        setattr(server, key, value)
    return server_mod, server


def test_the_echo_guard_covers_every_vad_gated_offline_backend():
    server_mod, server = _server()
    guard = server_mod.LocalAIServer._stt_is_vad_gated_offline
    assert guard(SimpleNamespace(stt_backend="onnx_asr")) is True
    assert guard(SimpleNamespace(stt_backend="sherpa", sherpa_model_type="offline")) is True
    assert guard(SimpleNamespace(stt_backend="sherpa", sherpa_model_type="online")) is False
    assert guard(SimpleNamespace(stt_backend="tone")) is False
    assert server._stt_is_available() is False
    server.onnx_asr_backend = _FakeServerBackend()
    assert server._stt_is_available() is True


@pytest.mark.asyncio
async def test_audio_flows_through_a_per_session_vad_to_final_transcripts():
    server_mod, server = _server()
    backend = _FakeServerBackend(results=[None, {"type": "final", "text": "  Здравствуйте  "}])
    server.onnx_asr_backend = backend
    session = server_mod.SessionContext(call_id="call-1")
    frame = b"\x01\x00" * 320  # 20 ms at 16 kHz

    assert await server._process_stt_stream_onnx_asr(session, frame, 16_000) == []
    assert session.onnx_asr_vad is not None
    # The session's stream memory is created with the first chunk and bound to the new VAD
    # before that chunk is fed, so the VAD's sample 0 is known in the stream.
    context = session.stt_context
    assert context is backend.contexts[0] and context.bound == 1
    assert backend.process_calls[0] == (session.onnx_asr_vad, frame, context)

    updates = await server._process_stt_stream_onnx_asr(session, frame, 16_000)
    assert updates == [{"text": "Здравствуйте", "is_final": True, "is_partial": False, "confidence": None}]
    assert len(backend.process_calls) == 2
    vad, pcm16, passed = backend.process_calls[1]
    assert vad is session.onnx_asr_vad and pcm16 == frame and passed is context
    # A reset drops the VAD but keeps the stream memory: the next phrase's pre-roll is
    # the audio that really preceded it. The VAD created for the next chunk is bound to it.
    server._reset_stt_session(session, "")
    assert session.onnx_asr_vad is None and session.stt_context is context
    await server._process_stt_stream_onnx_asr(session, frame, 16_000)
    assert session.onnx_asr_vad is not None
    assert backend.contexts == [context] and context.bound == 2


@pytest.mark.asyncio
async def test_trailing_speech_is_flushed_before_the_agent_speaks():
    from unittest.mock import AsyncMock

    server_mod, server = _server()
    backend = _FakeServerBackend(trailing={"type": "final", "text": "до свидания"})
    server.onnx_asr_backend = backend
    server._emit_stt_result = AsyncMock()
    session = server_mod.SessionContext(call_id="call-2")
    session.last_request_meta = {"mode": "stt", "request_id": "req-9"}

    await server._flush_onnx_asr_trailing(object(), session)  # no VAD yet: nothing to flush
    server._emit_stt_result.assert_not_awaited()

    session.onnx_asr_vad = object()
    session.stt_context = _FakeContext()
    await server._flush_onnx_asr_trailing("ws", session)
    assert backend.finalize_calls == [(session.onnx_asr_vad, session.stt_context)]
    server._emit_stt_result.assert_awaited_once()
    args, kwargs = server._emit_stt_result.await_args
    assert args[:3] == ("ws", "до свидания", session) and args[3] == "req-9"
    assert kwargs == {"source_mode": "stt", "is_final": True, "is_partial": False, "confidence": None}


@pytest.mark.asyncio
async def test_eight_khz_clients_are_upsampled_by_the_fir_by_default_and_by_ratecv_on_request():
    """The engine sends 16 kHz; a client that still sends 8 kHz gets the per-session FIR upsampler."""
    server_mod, server = _server()
    server.audio_processor = server_mod.AudioProcessor()
    frame_8k = np.rint(8000 * np.sin(2 * np.pi * 1000 * np.arange(1600) / 8000)).astype(np.int16).tobytes()

    session = server_mod.SessionContext(call_id="call-8k")
    assert await server._offline_stt_ingress(session, frame_8k, 16_000) is frame_8k  # already 16 kHz
    upsampled = await server._offline_stt_ingress(session, frame_8k, 8000)
    assert len(upsampled) == 2 * len(frame_8k)
    assert isinstance(session.stt_upsampler, server_mod.FirUpsampler)
    upsampler = session.stt_upsampler
    await server._offline_stt_ingress(session, frame_8k, 8000)
    assert session.stt_upsampler is upsampler  # one stateful upsampler per session

    server.config.local_stt_resampler = "ratecv"
    session = server_mod.SessionContext(call_id="call-8k-legacy")
    legacy = await server._offline_stt_ingress(session, frame_8k, 8000)
    assert abs(len(legacy) - 2 * len(frame_8k)) <= 4 and session.stt_upsampler is None  # ratecv drops a sample


@pytest.mark.asyncio
async def test_loading_creates_the_backend_from_the_config_and_releases_it_on_switch(monkeypatch):
    from unittest.mock import AsyncMock

    server_mod, server = _server(
        onnx_asr_model="gigaam-v3-rnnt",
        onnx_asr_model_path="",
        onnx_asr_cache_dir="/app/models/stt/onnx-asr",
        onnx_asr_quantization="int8",
        onnx_asr_device="auto",
    )
    created = {}

    class _FakeBackendClass(_FakeServerBackend):
        def __init__(self, **kwargs):
            super().__init__()
            created.update(kwargs)

        def initialize(self):
            return True

    stt_backends = _load("stt_backends")
    monkeypatch.setattr(stt_backends, "OnnxAsrSTTBackend", _FakeBackendClass)
    server._ensure_silero_vad_model = AsyncMock(return_value="/app/models/vad/silero_vad.onnx")

    await server._load_stt_model()

    assert isinstance(server.onnx_asr_backend, _FakeBackendClass)
    server._ensure_silero_vad_model.assert_awaited_once_with(log_tag="ONNX-ASR")
    assert created == {
        "model": "gigaam-v3-rnnt",
        "vad_model_path": "/app/models/vad/silero_vad.onnx",
        "model_path": "",
        "cache_dir": "/app/models/stt/onnx-asr",
        "quantization": "int8",
        "device": "auto",
        "sample_rate": 16_000,
        "preroll_ms": 350,
        "vad_threshold": 0.35,
        "vad_min_silence_ms": 700,
        "vad_min_speech_ms": 200,
        "postroll_ms": 300,
        "normalize_dbfs": -20.0,
        "normalize_max_gain_db": 24.0,
    }
    assert server.startup_errors == {}

    # Switching to another backend releases the model and its GPU memory.
    loaded = server.onnx_asr_backend
    server.stt_backend = "tone"
    server._load_tone_backend = AsyncMock()
    await server._load_stt_model()
    assert loaded.shutdown_calls == 1 and server.onnx_asr_backend is None
    server._load_tone_backend.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_backend_that_fails_to_initialize_is_reported_not_fatal(monkeypatch):
    from unittest.mock import AsyncMock

    server_mod, server = _server(
        onnx_asr_model="gigaam-v3-e2e-ctc",
        onnx_asr_model_path="",
        onnx_asr_cache_dir="/app/models/stt/onnx-asr",
        onnx_asr_quantization="",
        onnx_asr_device="cuda",
    )

    class _Broken(_FakeServerBackend):
        def __init__(self, **kwargs):
            super().__init__()

        def initialize(self):
            return False

    monkeypatch.setattr(_load("stt_backends"), "OnnxAsrSTTBackend", _Broken)
    server._ensure_silero_vad_model = AsyncMock(return_value="/vad.onnx")

    await server._load_stt_model()

    assert server.onnx_asr_backend is None
    assert "gigaam-v3-e2e-ctc" in server.startup_errors["stt"]


# --- whole utterances cut by the client's VAD -----------------------------------------


def test_a_whole_utterance_is_decoded_as_it_is_and_brought_to_the_loudness_target():
    backend = _ready_backend("три комнаты", normalize_dbfs=-20.0, normalize_max_gain_db=24.0)
    quiet = (0.01 * np.sqrt(2) * np.sin(2 * np.pi * 300 * np.arange(8000) / 16_000)).astype(np.float32)  # -40 dBFS, 500 ms
    pcm16 = (quiet * 32767).astype(np.int16).tobytes()

    result = backend.transcribe_utterance(pcm16)

    assert result == {"type": "final", "text": "три комнаты"}
    waveform, sample_rate = backend.recognizer.calls[0]
    assert sample_rate == 16_000 and len(waveform) == 8000
    assert 20 * np.log10(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) == pytest.approx(-20.0, abs=0.2)


def test_an_utterance_shorter_than_the_decode_floor_is_reported_empty_not_guessed():
    backend = _ready_backend("да", vad_min_speech_ms=200)
    assert backend.transcribe_utterance(_pcm16(0.1, 16_000 * 100 // 1000)) == {"type": "final", "text": ""}
    assert backend.recognizer.calls == []


def test_a_very_long_utterance_is_decoded_in_pieces():
    backend = _ready_backend("кусок")
    seconds = 45
    pcm16 = _pcm16(0.1, 16_000 * seconds)

    result = backend.transcribe_utterance(pcm16)

    assert result == {"type": "final", "text": "кусок кусок кусок"}  # 20 s + 20 s + 5 s
    assert [len(w) for w, _ in backend.recognizer.calls] == [320_000, 320_000, 80_000]


def test_an_uninitialized_backend_decodes_no_utterance():
    backend = _backend()
    assert backend.transcribe_utterance(_pcm16(0.1, 16_000)) is None
