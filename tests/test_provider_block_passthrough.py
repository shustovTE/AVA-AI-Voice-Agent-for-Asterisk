"""Unknown provider-block keys are the endpoint's, not the engine's.

A provider block describes one endpoint, so a key the engine has no meaning
for is meant for the API. Engine settings, identity metadata and credentials
must never make that trip.
"""
import pytest

from src.config import OpenAIProviderConfig, split_openai_passthrough_fields
from src.pipelines.orchestrator import _with_openai_passthrough

PROVIDER_BLOCK = {
    "type": "openai",
    "name": "native_llm",
    "enabled": True,
    "capabilities": ["llm"],
    "api_key": "secret-value",
    "api_key_file": "/app/project/secrets/providers/native_llm/api-key",
    "chat_base_url": "https://api.mistral.ai/v1",
    "chat_model": "mistral-small-latest",
    "response_timeout_sec": 15,
    "temperature": 0.7,
    "max_tokens": 150,
    "prompt_cache_key": "prompt-1",
    "safe_prompt": True,
}


def test_vendor_fields_are_collected():
    passthrough = split_openai_passthrough_fields(PROVIDER_BLOCK)
    assert passthrough["prompt_cache_key"] == "prompt-1"
    assert passthrough["safe_prompt"] is True


@pytest.mark.parametrize(
    "key",
    [
        "type",
        "name",
        "enabled",
        "capabilities",
        "chat_base_url",
        "chat_model",
        "response_timeout_sec",
    ],
)
def test_engine_settings_stay_home(key):
    assert key not in split_openai_passthrough_fields(PROVIDER_BLOCK)


@pytest.mark.parametrize(
    "key", ["api_key", "api_key_file", "api_key_env", "auth_token", "client_secret", "password"]
)
def test_credentials_are_never_forwarded(key):
    block = {**PROVIDER_BLOCK, key: "do-not-send"}
    assert key not in split_openai_passthrough_fields(block)


def test_chat_completions_parameters_are_forwarded():
    # Neither is a field of the typed config, and both are real API parameters.
    # `max_tokens` also guards the credential match: it must not read as a token.
    passthrough = split_openai_passthrough_fields(PROVIDER_BLOCK)
    assert passthrough["temperature"] == 0.7
    assert passthrough["max_tokens"] == 150


def test_null_values_are_skipped():
    assert "prompt_cache_key" not in split_openai_passthrough_fields(
        {"prompt_cache_key": None}
    )


def test_hydration_moves_bare_keys_into_extra_body():
    config = OpenAIProviderConfig(**_with_openai_passthrough(PROVIDER_BLOCK))
    assert config.extra_body["prompt_cache_key"] == "prompt-1"
    assert "api_key" not in config.extra_body
    assert config.chat_model == "mistral-small-latest"


def test_explicit_extra_body_wins_over_a_bare_key():
    block = {
        **PROVIDER_BLOCK,
        "extra_body": {"prompt_cache_key": "explicit"},
    }
    config = OpenAIProviderConfig(**_with_openai_passthrough(block))
    assert config.extra_body["prompt_cache_key"] == "explicit"
    assert config.extra_body["safe_prompt"] is True


def test_block_without_vendor_fields_gets_no_extra_body():
    block = {
        "type": "openai",
        "name": "openai_llm",
        "chat_model": "gpt-4o-mini",
        "api_key": "k",
    }
    assert _with_openai_passthrough(block).get("extra_body") in (None, {})
