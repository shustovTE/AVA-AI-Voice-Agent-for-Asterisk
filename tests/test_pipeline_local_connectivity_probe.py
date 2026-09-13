"""The local connectivity probe dials the address the adapter actually uses.

The generic validator receives a pipeline's role options alone, so a ws_url
configured on the provider was invisible to it and every startup check reported
a connection failure against a hardcoded loopback address while calls worked.
"""
import pytest

from src.config import AppConfig, LocalProviderConfig
from src.pipelines.local import LocalLLMAdapter, LocalSTTAdapter, LocalTTSAdapter


def _app_config() -> AppConfig:
    return AppConfig(
        default_provider="local",
        providers={"local": {"enabled": True}},
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "prompt", "model": "gpt-4o"},
        audio_transport="audiosocket",
    )


def _adapter(cls, key, provider_config=None, pipeline_defaults=None):
    return cls(
        key,
        _app_config(),
        provider_config or LocalProviderConfig(),
        pipeline_defaults or {},
    )


class _ProbeRecorder:
    """Stand in for the websocket reachability check.

    A class instance rather than a function: patched onto Component it stays a
    plain attribute, so it is called without an implicit self.
    """

    def __init__(self, healthy=True):
        self.urls = []
        self._healthy = healthy

    async def __call__(self, url, api_key=None, timeout=5.0):
        self.urls.append(url)
        if self._healthy:
            return {"healthy": True, "error": None, "details": {"endpoint": url}}
        return {"healthy": False, "error": "Connection refused", "details": {}}


@pytest.fixture
def probe(monkeypatch):
    recorder = _ProbeRecorder()
    monkeypatch.setattr(
        "src.pipelines.base.Component._test_websocket_connection", recorder
    )
    return recorder


@pytest.mark.asyncio
async def test_provider_ws_url_is_probed(probe):
    """The address lives on the provider, and the pipeline role never sees it."""
    adapter = _adapter(
        LocalSTTAdapter,
        "local_stt",
        LocalProviderConfig(ws_url="ws://local-ai-server:8765"),
    )

    result = await adapter.validate_connectivity({})

    assert probe.urls == ["ws://local-ai-server:8765"]
    assert result["healthy"] is True


@pytest.mark.asyncio
async def test_pipeline_options_override_the_provider(probe):
    adapter = _adapter(
        LocalSTTAdapter,
        "local_stt",
        LocalProviderConfig(ws_url="ws://from-provider:8765"),
        {"ws_url": "ws://from-pipeline:9000"},
    )

    await adapter.validate_connectivity({})

    assert probe.urls == ["ws://from-pipeline:9000"]


@pytest.mark.asyncio
async def test_runtime_options_win_over_both(probe):
    adapter = _adapter(
        LocalSTTAdapter,
        "local_stt",
        LocalProviderConfig(ws_url="ws://from-provider:8765"),
        {"ws_url": "ws://from-pipeline:9000"},
    )

    await adapter.validate_connectivity({"ws_url": "ws://from-runtime:7000"})

    assert probe.urls == ["ws://from-runtime:7000"]


@pytest.mark.asyncio
async def test_a_cleared_ws_url_falls_back_to_the_adapter_default(probe):
    """An explicit null used to survive setdefault and reach the probe."""
    adapter = _adapter(LocalSTTAdapter, "local_stt", LocalProviderConfig(ws_url=None))

    await adapter.validate_connectivity({})

    assert probe.urls == ["ws://127.0.0.1:8765"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls,key",
    [
        (LocalSTTAdapter, "local_stt"),
        (LocalLLMAdapter, "local_llm"),
        (LocalTTSAdapter, "local_tts"),
    ],
)
async def test_every_local_role_probes_the_configured_address(probe, cls, key):
    adapter = _adapter(cls, key, LocalProviderConfig(ws_url="ws://local-ai:8765"))

    await adapter.validate_connectivity({})

    assert probe.urls == ["ws://local-ai:8765"]


@pytest.mark.asyncio
async def test_probe_failure_is_still_reported(monkeypatch):
    monkeypatch.setattr(
        "src.pipelines.base.Component._test_websocket_connection",
        _ProbeRecorder(healthy=False),
    )
    adapter = _adapter(LocalSTTAdapter, "local_stt")

    result = await adapter.validate_connectivity({})

    assert result["healthy"] is False


def test_compose_options_resolves_a_cleared_ws_url():
    adapter = _adapter(LocalSTTAdapter, "local_stt", LocalProviderConfig(ws_url=None))

    assert adapter._compose_options({})["ws_url"] == "ws://127.0.0.1:8765"
