"""`prompt_cache_key` is the one vendor field the LLM payload may carry.

Prompt caching on OpenAI-compatible endpoints (OpenAI, Mistral) is opt-in: the
key has to travel in the request body, and the payload builder otherwise emits
a fixed set of fields, so a value configured on the provider had no way out.
"""
import pytest

from src.config import AppConfig, OpenAIProviderConfig
from src.pipelines.openai import OpenAILLMAdapter


def _app_config() -> AppConfig:
    return AppConfig(
        default_provider="native_llm",
        providers={"native_llm": {"api_key": "test-key"}},
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "persona", "model": "gpt-4o"},
        audio_transport="audiosocket",
        downstream_mode="stream",
    )


def _payload(provider_key=None, pipeline_options=None, runtime_options=None):
    provider = OpenAIProviderConfig(api_key="test-key", prompt_cache_key=provider_key)
    adapter = OpenAILLMAdapter(
        "native_llm", _app_config(), provider, pipeline_options or {}
    )
    merged = adapter._compose_options(runtime_options or {})
    return adapter._build_chat_payload("привет", {}, merged)


def test_provider_key_reaches_the_request_body():
    assert _payload(provider_key="prompt-1")["prompt_cache_key"] == "prompt-1"


def test_pipeline_options_override_the_provider():
    payload = _payload(
        provider_key="prompt-1",
        pipeline_options={"prompt_cache_key": "prompt-1.1"},
    )
    assert payload["prompt_cache_key"] == "prompt-1.1"


def test_runtime_options_win():
    payload = _payload(
        provider_key="prompt-1",
        pipeline_options={"prompt_cache_key": "prompt-1.1"},
        runtime_options={"prompt_cache_key": "prompt-2"},
    )
    assert payload["prompt_cache_key"] == "prompt-2"


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_unset_key_sends_no_field(configured):
    payload = _payload(provider_key=configured)
    assert "prompt_cache_key" not in payload
    assert set(payload) == {"model", "messages", "temperature"}


def test_non_string_value_is_sent_as_text():
    # YAML turns `prompt_cache_key: 1` into an int; the API expects a string.
    payload = _payload(pipeline_options={"prompt_cache_key": 1})
    assert payload["prompt_cache_key"] == "1"


def test_unknown_provider_keys_are_still_dropped():
    provider = OpenAIProviderConfig(api_key="k", **{"CACHE_KEY": "prompt-1"})
    assert not hasattr(provider, "CACHE_KEY")
