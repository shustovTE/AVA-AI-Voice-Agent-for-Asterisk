"""ElevenLabs requests can be routed through an HTTP proxy on their own.

The deployment this serves keeps engine, models and Asterisk in one region and
tunnels only the one foreign leg, so the proxy must apply to this adapter and
to nothing else.
"""
import base64

import pytest

from src.config import ElevenLabsProviderConfig
from src.pipelines.elevenlabs import (  # noqa: F401
    ElevenLabsTTSAdapter,
    sanitize_proxy_url,
    split_proxy_credentials,
)
from tests.test_pipeline_elevenlabs_streaming import (
    _FakeAudioResponse,
    _FakeHttpSession,
    _adapter,
    _app_config,
)

MULAW_FORMAT = {"format": {"encoding": "mulaw", "sample_rate": 8000}}


async def _synthesize(adapter):
    return [chunk async for chunk in adapter.synthesize("call-1", "привет", MULAW_FORMAT)]


def _session():
    return _FakeHttpSession(_FakeAudioResponse([bytes([0x7F]) * 160]))


# --- URL handling ------------------------------------------------------------


def test_no_proxy_configured_sends_nothing():
    assert split_proxy_credentials(None) == (None, None)
    assert split_proxy_credentials("") == (None, None)
    assert split_proxy_credentials("   ") == (None, None)


def test_plain_proxy_url_passes_through():
    assert split_proxy_credentials("http://xray:8080") == ("http://xray:8080", None)


def test_inline_credentials_become_a_proxy_authorization_header():
    url, headers = split_proxy_credentials("http://bob:s3cret@xray:8080")

    assert url == "http://xray:8080"
    expected = base64.b64encode(b"bob:s3cret").decode()
    assert headers == {"Proxy-Authorization": f"Basic {expected}"}


def test_percent_encoded_credentials_are_decoded():
    _, headers = split_proxy_credentials("http://bob:p%40ss%3Aword@xray:8080")

    expected = base64.b64encode(b"bob:p@ss:word").decode()
    assert headers == {"Proxy-Authorization": f"Basic {expected}"}


def test_socks_scheme_is_rejected_with_a_usable_message():
    with pytest.raises(ValueError) as excinfo:
        split_proxy_credentials("socks5://xray:1080")

    message = str(excinfo.value)
    assert "socks5" in message
    assert "aiohttp-socks" in message


def test_proxy_without_a_host_is_rejected():
    with pytest.raises(ValueError):
        split_proxy_credentials("http://")


def test_sanitize_strips_credentials_for_logging():
    assert sanitize_proxy_url("http://bob:s3cret@xray:8080") == "http://xray:8080"
    assert sanitize_proxy_url("http://xray:8080") == "http://xray:8080"


# --- adapter wiring ----------------------------------------------------------


@pytest.mark.asyncio
async def test_requests_carry_no_proxy_by_default():
    session = _session()
    adapter = _adapter(session)

    await _synthesize(adapter)

    assert session.requests[0]["proxy"] is None
    assert session.requests[0]["proxy_headers"] is None


@pytest.mark.asyncio
async def test_provider_proxy_reaches_the_request():
    session = _session()
    adapter = _adapter(
        session,
        provider_config=ElevenLabsProviderConfig(
            api_key="test-key", proxy="http://xray:8080"
        ),
    )

    await _synthesize(adapter)

    assert session.requests[0]["proxy"] == "http://xray:8080"
    assert session.requests[0]["proxy_headers"] is None


@pytest.mark.asyncio
async def test_authenticated_proxy_reaches_the_request():
    session = _session()
    adapter = _adapter(
        session,
        provider_config=ElevenLabsProviderConfig(
            api_key="test-key", proxy="http://bob:s3cret@xray:8080"
        ),
    )

    await _synthesize(adapter)

    request = session.requests[0]
    assert request["proxy"] == "http://xray:8080"
    assert "Proxy-Authorization" in request["proxy_headers"]


@pytest.mark.asyncio
async def test_pipeline_options_override_the_provider_proxy():
    session = _session()
    adapter = _adapter(
        session,
        provider_config=ElevenLabsProviderConfig(
            api_key="test-key", proxy="http://provider:8080"
        ),
        options={"proxy": "http://pipeline:3128"},
    )

    await _synthesize(adapter)

    assert session.requests[0]["proxy"] == "http://pipeline:3128"


@pytest.mark.asyncio
async def test_pipeline_options_can_disable_the_provider_proxy():
    session = _session()
    adapter = _adapter(
        session,
        provider_config=ElevenLabsProviderConfig(
            api_key="test-key", proxy="http://provider:8080"
        ),
        options={"proxy": ""},
    )

    await _synthesize(adapter)

    assert session.requests[0]["proxy"] is None


def test_a_bad_proxy_fails_the_adapter_instead_of_going_direct():
    with pytest.raises(ValueError):
        _adapter(
            _session(),
            provider_config=ElevenLabsProviderConfig(
                api_key="test-key", proxy="socks5://xray:1080"
            ),
        )


# --- connection reuse --------------------------------------------------------


def test_keepalive_is_unset_by_default():
    adapter = _adapter(_session())

    assert adapter._keepalive_timeout_sec is None


def test_keepalive_is_read_from_the_provider():
    adapter = _adapter(
        _session(),
        provider_config=ElevenLabsProviderConfig(
            api_key="test-key", keepalive_timeout_sec=120
        ),
    )

    assert adapter._keepalive_timeout_sec == pytest.approx(120.0)


def test_pipeline_options_override_the_keepalive():
    adapter = _adapter(
        _session(),
        provider_config=ElevenLabsProviderConfig(
            api_key="test-key", keepalive_timeout_sec=120
        ),
        options={"keepalive_timeout_sec": "45"},
    )

    assert adapter._keepalive_timeout_sec == pytest.approx(45.0)


def test_unusable_keepalive_values_fall_back_to_the_aiohttp_default():
    for value in ("soon", 0, -5, []):
        adapter = _adapter(
            _session(),
            provider_config=ElevenLabsProviderConfig(api_key="test-key"),
            options={"keepalive_timeout_sec": value},
        )

        assert adapter._keepalive_timeout_sec is None


@pytest.mark.asyncio
async def test_the_real_session_carries_the_configured_keepalive():
    """Without an injected factory the adapter builds its own connector."""
    adapter = ElevenLabsTTSAdapter(
        "elevenlabs_tts",
        _app_config(),
        ElevenLabsProviderConfig(api_key="test-key", keepalive_timeout_sec=90),
        {},
    )
    try:
        await adapter._ensure_session()

        assert adapter._session.connector._keepalive_timeout == pytest.approx(90.0)
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_the_real_session_keeps_the_aiohttp_default_when_unset():
    adapter = ElevenLabsTTSAdapter(
        "elevenlabs_tts",
        _app_config(),
        ElevenLabsProviderConfig(api_key="test-key"),
        {},
    )
    try:
        await adapter._ensure_session()

        assert adapter._session.connector._keepalive_timeout == pytest.approx(15.0)
    finally:
        await adapter.stop()


# --- a modular single-capability provider block ------------------------------


def _modular_app_config(provider_block):
    """One provider carrying only the tts capability, as the Admin UI writes it."""
    from src.config import AppConfig

    return AppConfig(
        default_provider="local",
        providers={
            "local": {"enabled": True},
            "elevenlabs_tts": {
                "type": "elevenlabs",
                "capabilities": ["tts"],
                "api_key": "test-key",
                **provider_block,
            },
        },
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "prompt", "model": "gpt-4o"},
        pipelines={"p": {"stt": "local_stt", "llm": "local_llm", "tts": "elevenlabs_tts"}},
        active_pipeline="p",
        audio_transport="audiosocket",
    )


@pytest.mark.asyncio
async def test_modular_provider_block_carries_the_proxy_to_the_adapter():
    """The single-capability editor writes this shape, not the full-agent one."""
    from src.pipelines.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(
        _modular_app_config({"proxy": "http://xray:8080", "keepalive_timeout_sec": 120})
    )
    await orchestrator.start()

    resolution = orchestrator.get_pipeline("call-1")
    adapter = resolution.tts_adapter

    assert isinstance(adapter, ElevenLabsTTSAdapter)
    assert adapter._proxy_url == "http://xray:8080"
    assert adapter._keepalive_timeout_sec == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_modular_provider_block_without_routing_stays_direct():
    from src.pipelines.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(_modular_app_config({}))
    await orchestrator.start()

    adapter = orchestrator.get_pipeline("call-2").tts_adapter

    assert adapter._proxy_url is None
    assert adapter._keepalive_timeout_sec is None


@pytest.mark.asyncio
async def test_an_emptied_proxy_field_stays_direct():
    """Clearing the field in the editor saves an empty string, not a removal."""
    from src.pipelines.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(
        _modular_app_config({"proxy": "", "keepalive_timeout_sec": ""})
    )
    await orchestrator.start()

    adapter = orchestrator.get_pipeline("call-3").tts_adapter

    assert adapter._proxy_url is None
    assert adapter._keepalive_timeout_sec is None
