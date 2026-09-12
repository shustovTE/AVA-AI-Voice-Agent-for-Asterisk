"""Vendor-specific Chat Completions fields reach the request body.

`prompt_cache_key` (OpenAI/Mistral prompt caching) and vLLM's
`chat_template_kwargs` have no first-class option, and the payload builder
only emits a fixed set of keys, so without a pass-through they were silently
dropped from the config.
"""
import pytest

from src.config import AppConfig, OpenAIProviderConfig
from src.pipelines import openai as openai_module
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


def _payload(pipeline_options=None, provider_extra=None, runtime_options=None):
    provider = OpenAIProviderConfig(
        api_key="test-key", extra_body=provider_extra or {}
    )
    adapter = OpenAILLMAdapter(
        "native_llm", _app_config(), provider, pipeline_options or {}
    )
    merged = adapter._compose_options(runtime_options or {})
    return adapter._build_chat_payload("привет", {}, merged)


def test_pipeline_extra_body_reaches_the_request():
    payload = _payload({"extra_body": {"prompt_cache_key": "prompt-1"}})
    assert payload["prompt_cache_key"] == "prompt-1"


def test_provider_and_pipeline_extra_body_merge_per_key():
    payload = _payload(
        pipeline_options={"extra_body": {"prompt_cache_key": "prompt-1.1"}},
        provider_extra={"prompt_cache_key": "prompt-1", "safe_prompt": True},
    )
    assert payload["prompt_cache_key"] == "prompt-1.1"
    assert payload["safe_prompt"] is True


def test_runtime_options_win_over_pipeline():
    payload = _payload(
        pipeline_options={"extra_body": {"prompt_cache_key": "pipeline"}},
        runtime_options={"extra_body": {"prompt_cache_key": "runtime"}},
    )
    assert payload["prompt_cache_key"] == "runtime"


def test_nested_vendor_structures_survive():
    payload = _payload(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("key", ["model", "messages", "stream", "tools", "tool_choice"])
def test_engine_owned_keys_cannot_be_overridden(key):
    payload = _payload({"extra_body": {key: "hijacked"}})
    assert payload.get(key) != "hijacked"
    assert payload["model"]
    assert payload["messages"]


def test_absent_extra_body_leaves_the_payload_unchanged():
    payload = _payload()
    assert set(payload) == {"model", "messages", "temperature"}


def test_forwarded_keys_are_logged(monkeypatch):
    """The operator must be able to see what leaves for the endpoint.

    structlog does not flow reliably through pytest's caplog, so patch the
    logger the module actually uses (same approach as the realtime tests).
    """
    calls = []
    monkeypatch.setattr(
        openai_module.logger,
        "info",
        lambda event, **kw: calls.append((event, kw)),
    )

    _payload({"extra_body": {"prompt_cache_key": "prompt-1", "safe_prompt": True}})

    assert any(
        kw.get("keys") == ["prompt_cache_key", "safe_prompt"]
        for _event, kw in calls
    )


def test_values_are_not_logged(monkeypatch):
    calls = []
    monkeypatch.setattr(
        openai_module.logger,
        "info",
        lambda event, **kw: calls.append((event, kw)),
    )

    _payload({"extra_body": {"prompt_cache_key": "prompt-1"}})

    assert not any("prompt-1" in str(kw) for _event, kw in calls)


def test_unknown_provider_keys_are_still_dropped():
    provider = OpenAIProviderConfig(api_key="k", **{"CACHE_KEY": "prompt-1"})
    assert not hasattr(provider, "CACHE_KEY")
