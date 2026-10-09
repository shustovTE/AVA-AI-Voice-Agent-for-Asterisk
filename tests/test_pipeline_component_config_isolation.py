import pytest

from src.config import AppConfig
from src.pipelines.orchestrator import PipelineOrchestrator, PipelineOrchestratorError
from src.pipelines.local import LocalTTSAdapter
from src.pipelines.openai import OpenAILLMAdapter, OpenAISTTAdapter, OpenAITTSAdapter


def _config(providers, *, stt="local_stt", llm="local_llm", tts="local_tts"):
    return AppConfig(
        default_provider="local",
        providers=providers,
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "prompt"},
        pipelines={"test": {"stt": stt, "llm": llm, "tts": tts}},
        active_pipeline="test",
    )


def test_local_component_blocks_are_hydrated_independently():
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": {"enabled": True, "ws_url": "ws://base:8765"},
                "local_stt": {"enabled": True, "ws_url": "ws://stt:8765"},
                "local_llm": {"enabled": False, "ws_url": "ws://llm:8765"},
                "local_tts": {"enabled": True, "ws_url": "ws://tts:8765"},
            }
        )
    )

    configs = orchestrator._local_component_configs
    assert configs["local_stt"].effective_ws_url == "ws://stt:8765"
    assert configs["local_tts"].effective_ws_url == "ws://tts:8765"
    assert "local_llm" not in configs

    with pytest.raises(PipelineOrchestratorError, match="local_llm"):
        orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])


def test_base_local_block_remains_a_compatibility_fallback_for_all_roles():
    orchestrator = PipelineOrchestrator(
        _config({"local": {"enabled": True, "ws_url": "ws://compat:8765"}})
    )

    assert set(orchestrator._local_component_configs) == {
        "local_stt",
        "local_llm",
        "local_tts",
    }
    assert {
        config.effective_ws_url
        for config in orchestrator._local_component_configs.values()
    } == {"ws://compat:8765"}


def test_openai_component_blocks_do_not_overwrite_each_other():
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "openai": {"api_key": "test-key"},
                "openai_stt": {
                    "enabled": True,
                    "stt_base_url": "https://stt.example/v1/audio/transcriptions",
                    "stt_model": "stt-model",
                },
                "openai_llm": {
                    "enabled": False,
                    "chat_base_url": "https://llm.example/v1",
                    "chat_model": "llm-model",
                },
                "openai_tts": {
                    "enabled": True,
                    "tts_base_url": "https://tts.example/v1/audio/speech",
                    "tts_model": "tts-model",
                },
            },
            stt="openai_stt",
            llm="openai_llm",
            tts="openai_tts",
        )
    )

    configs = orchestrator._openai_component_configs
    assert configs["openai_stt"].stt_base_url == "https://stt.example/v1/audio/transcriptions"
    assert configs["openai_stt"].stt_model == "stt-model"
    assert configs["openai_tts"].tts_base_url == "https://tts.example/v1/audio/speech"
    assert configs["openai_tts"].tts_model == "tts-model"
    assert "openai_llm" not in configs

    with pytest.raises(PipelineOrchestratorError, match="openai_llm"):
        orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])


def test_custom_openai_compatible_llm_reads_provider_scoped_key_file(tmp_path):
    key_file = tmp_path / "deepseek-key"
    key_file.write_text("deepseek-secret")
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "deepseek_llm": {
                    "type": "openai",
                    "enabled": True,
                    "api_key_file": str(key_file),
                    "chat_base_url": "https://api.deepseek.com/v1",
                    "chat_model": "deepseek-chat",
                }
            },
            llm="deepseek_llm",
        )
    )

    adapter = orchestrator._build_component("deepseek_llm", {})
    assert isinstance(adapter, OpenAILLMAdapter)
    assert adapter._provider_defaults.api_key == "deepseek-secret"
    assert adapter._provider_defaults.chat_model == "deepseek-chat"


_LOCAL = {"enabled": True, "ws_url": "ws://local:8765"}


def test_custom_openai_compatible_tts_block_is_registered_under_its_own_key():
    """A `<name>_tts` block of type openai drives the OpenAI TTS adapter under its own key.

    Only `*_llm` blocks used to be registered by their own key; a self-hosted
    speech endpoint saved as `fish_tts` resolved to the wildcard placeholder
    and the pipeline was refused at startup.
    """
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "fish_tts": {
                    "type": "openai",
                    "api_key": "local",
                    "tts_base_url": "http://tts.lan:8091/v1/audio/speech",
                    "tts_model": "fishaudio/s2-pro",
                    "voice": "anna",
                    "tts_response_format": "pcm",
                    "tts_streaming": True,
                    "tts_pcm_sample_rate_hz": 44100,
                    "tts_text_prefix": "<|speaker:0|>",
                    "tts_extra_body": {"stream_format": "audio"},
                },
            },
            tts="fish_tts",
        )
    )

    orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])
    adapter = orchestrator._build_component("fish_tts", {})
    assert isinstance(adapter, OpenAITTSAdapter)
    defaults = adapter._provider_defaults
    assert defaults.api_key == "local"
    assert defaults.tts_base_url == "http://tts.lan:8091/v1/audio/speech"
    assert defaults.tts_model == "fishaudio/s2-pro"
    assert defaults.voice == "anna"
    assert defaults.tts_response_format == "pcm"
    assert defaults.tts_streaming is True
    assert defaults.tts_pcm_sample_rate_hz == 44100
    assert defaults.tts_text_prefix == "<|speaker:0|>"
    assert defaults.tts_extra_body == {"stream_format": "audio"}


def test_custom_openai_compatible_stt_block_is_registered_under_its_own_key():
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "whisper_stt": {
                    "type": "openai",
                    "api_key": "local",
                    "stt_base_url": "http://stt.lan:8000/v1/audio/transcriptions",
                    "stt_model": "Systran/faster-whisper-large-v3",
                },
            },
            stt="whisper_stt",
        )
    )

    orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])
    adapter = orchestrator._build_component("whisper_stt", {})
    assert isinstance(adapter, OpenAISTTAdapter)
    assert adapter._provider_defaults.stt_base_url == "http://stt.lan:8000/v1/audio/transcriptions"
    assert adapter._provider_defaults.stt_model == "Systran/faster-whisper-large-v3"


def test_custom_local_tts_block_is_registered_under_its_own_key():
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "piper_tts": {"type": "local", "ws_url": "ws://piper:8765"},
            },
            tts="piper_tts",
        )
    )

    orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])
    adapter = orchestrator._build_component("piper_tts", {})
    assert isinstance(adapter, LocalTTSAdapter)


def test_canonical_openai_tts_block_keeps_its_base_block_merge():
    """`openai_tts` stays on the builtin path, so `providers.openai` still fills its gaps."""
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "openai": {"api_key": "base-key", "organization": "org-1"},
                "openai_tts": {"type": "openai", "tts_model": "tts-1-hd"},
            },
            tts="openai_tts",
        )
    )

    adapter = orchestrator._build_component("openai_tts", {})
    assert isinstance(adapter, OpenAITTSAdapter)
    assert adapter._provider_defaults.api_key == "base-key"
    assert adapter._provider_defaults.organization == "org-1"
    assert adapter._provider_defaults.tts_model == "tts-1-hd"


def test_custom_openai_compatible_speech_block_without_a_key_is_named_in_the_error(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "fish_tts": {
                    "type": "openai",
                    "tts_base_url": "http://tts.lan:8091/v1/audio/speech",
                },
            },
            tts="fish_tts",
        )
    )

    with pytest.raises(PipelineOrchestratorError) as excinfo:
        orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])
    message = str(excinfo.value)
    assert "cannot resolve tts component 'fish_tts'" in message
    assert "providers.fish_tts (type: openai) needs an api_key" in message
    assert "FISH_API_KEY" in message


def test_custom_speech_block_of_an_unsupported_type_is_named_in_the_error():
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "voice_tts": {"type": "deepgram", "api_key": "dg-key"},
            },
            tts="voice_tts",
        )
    )

    with pytest.raises(PipelineOrchestratorError, match="type 'deepgram', which is not registered as a tts component"):
        orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])


def test_disabled_custom_speech_block_is_named_in_the_error():
    orchestrator = PipelineOrchestrator(
        _config(
            {
                "local": dict(_LOCAL),
                "fish_tts": {
                    "type": "openai",
                    "enabled": False,
                    "api_key": "local",
                    "tts_base_url": "http://tts.lan:8091/v1/audio/speech",
                },
            },
            tts="fish_tts",
        )
    )

    with pytest.raises(PipelineOrchestratorError, match=r"providers\.fish_tts is disabled"):
        orchestrator._validate_pipeline_entry("test", orchestrator.config.pipelines["test"])
