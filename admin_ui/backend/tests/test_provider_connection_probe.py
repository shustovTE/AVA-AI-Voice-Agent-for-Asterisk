"""Provider "Test connection" probes must mirror what the engine actually does.

Two regressions are covered here: the Local AI probe never authenticated (so a
token-protected server reported "status invalid"), and the OpenAI-compatible
probe sent the operator's key to api.openai.com whenever the configured host
was not a known vendor (reporting a meaningless 401 for self-hosted vLLM).
"""
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

pytest.importorskip("fastapi")

import httpx  # noqa: E402
import websockets  # noqa: E402

from api import config as config_api  # noqa: E402
from api.config import ProviderTestRequest  # noqa: E402


class _FakeWebSocket:
    def __init__(self, replies):
        self._replies = list(replies)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    async def recv(self):
        if not self._replies:
            raise AssertionError("probe asked for more replies than expected")
        return self._replies.pop(0)


def _fake_connect(replies, seen):
    def connect(url, **_kwargs):
        seen.append(url)
        return _FakeWebSocket(replies)

    return connect


def _status_response(*, stt=True, llm=False, tts=False, runtime_mode="full"):
    import json

    return json.dumps(
        {
            "type": "status_response",
            "status": "ok",
            "stt_backend": "tone",
            "tts_backend": "piper",
            "models": {
                "stt": {"loaded": stt},
                "llm": {"loaded": llm, "path": ""},
                "tts": {"loaded": tts},
            },
            "config": {"runtime_mode": runtime_mode},
        }
    )


AUTH_OK = '{"type": "auth_response", "status": "ok"}'
AUTH_FAIL = '{"type": "auth_response", "status": "error", "message": "invalid_auth_token"}'


@pytest.mark.asyncio
async def test_local_probe_authenticates_and_scopes_to_declared_capabilities(
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(
        websockets, "connect", _fake_connect([AUTH_OK, _status_response()], seen)
    )

    result = await config_api.test_provider_connection(
        ProviderTestRequest(
            name="local_stt",
            config={
                "type": "local",
                "capabilities": ["stt"],
                "ws_url": "ws://172.17.0.1:8765",
                "auth_token": "secret-token",
            },
        )
    )

    assert seen == ["ws://172.17.0.1:8765"]
    assert result["success"] is True
    assert "STT: tone" in result["message"]


@pytest.mark.asyncio
async def test_local_probe_reports_rejected_token(monkeypatch):
    seen = []
    monkeypatch.setattr(websockets, "connect", _fake_connect([AUTH_FAIL], seen))

    result = await config_api.test_provider_connection(
        ProviderTestRequest(
            name="local_stt",
            config={
                "type": "local",
                "capabilities": ["stt"],
                "ws_url": "ws://172.17.0.1:8765",
                "auth_token": "wrong-token",
            },
        )
    )

    assert result["success"] is False
    assert "auth token" in result["message"].lower()


@pytest.mark.asyncio
async def test_local_probe_does_not_require_llm_in_minimal_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(
        websockets,
        "connect",
        _fake_connect(
            [_status_response(stt=True, llm=False, tts=True, runtime_mode="minimal")],
            seen,
        ),
    )

    result = await config_api.test_provider_connection(
        ProviderTestRequest(
            name="local",
            config={"type": "local", "ws_url": "ws://127.0.0.1:8765"},
        )
    )

    assert result["success"] is True


class _FakeHttpxClient:
    calls: list = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **kwargs):
        _FakeHttpxClient.calls.append((url, kwargs))
        return type(
            "Response",
            (),
            {"status_code": 200, "json": lambda self: {"data": [{"id": "m"}]}},
        )()


@pytest.mark.asyncio
async def test_openai_probe_calls_saved_self_hosted_endpoint(monkeypatch):
    _FakeHttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    monkeypatch.setattr(
        config_api,
        "_read_merged_config_dict",
        lambda: {
            "providers": {
                "native_llm": {
                    "type": "openai",
                    "chat_base_url": "https://ai-api.example/v1",
                }
            }
        },
    )

    result = await config_api.test_provider_connection(
        ProviderTestRequest(
            name="native_llm",
            config={
                "type": "openai",
                "chat_base_url": "https://ai-api.example/v1",
                "api_key": "self-hosted-token",
            },
        )
    )

    assert result["success"] is True
    url, kwargs = _FakeHttpxClient.calls[0]
    assert url == "https://ai-api.example/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer self-hosted-token"


@pytest.mark.asyncio
async def test_openai_probe_never_sends_the_key_to_an_unsaved_host(monkeypatch):
    _FakeHttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    monkeypatch.setattr(
        config_api,
        "_read_merged_config_dict",
        lambda: {
            "providers": {
                "native_llm": {
                    "type": "openai",
                    "chat_base_url": "https://ai-api.example/v1",
                }
            }
        },
    )

    result = await config_api.test_provider_connection(
        ProviderTestRequest(
            name="native_llm",
            config={
                "type": "openai",
                "chat_base_url": "https://attacker.example/v1",
                "api_key": "self-hosted-token",
            },
        )
    )

    assert result["success"] is False
    assert _FakeHttpxClient.calls == []


class _RecordingHttpxClient:
    """Records how the probe builds its client and lets a test script the outcome."""

    kwargs: list = []
    calls: list = []
    outcome = 200  # an HTTP status, or an exception instance to raise

    def __init__(self, **kwargs):
        _RecordingHttpxClient.kwargs.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **kwargs):
        _RecordingHttpxClient.calls.append((url, kwargs))
        outcome = _RecordingHttpxClient.outcome
        if isinstance(outcome, Exception):
            raise outcome
        return type(
            "Response",
            (),
            {"status_code": outcome, "json": lambda self: {"voices": [{"id": "a"}, {"id": "b"}]}},
        )()


def _reset_recording(outcome=200):
    _RecordingHttpxClient.kwargs = []
    _RecordingHttpxClient.calls = []
    _RecordingHttpxClient.outcome = outcome


ELEVENLABS_TTS = {
    "type": "elevenlabs",
    "capabilities": ["tts"],
    "api_key": "xi-test-key",
    "proxy": "http://user:secret@10.0.0.5:8080",
}


@pytest.mark.asyncio
async def test_elevenlabs_probe_takes_the_configured_proxy(monkeypatch):
    """A modular TTS provider of type elevenlabs is probed through its proxy."""
    _reset_recording()
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingHttpxClient)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="eleven_tts", config=dict(ELEVENLABS_TTS))
    )

    assert result["success"] is True
    assert "via proxy http://10.0.0.5:8080" in result["message"]
    assert "secret" not in result["message"]
    assert result["proxy"] == "http://10.0.0.5:8080"
    client_kwargs = _RecordingHttpxClient.kwargs[0]
    assert client_kwargs["trust_env"] is False
    proxy = client_kwargs["proxy"]
    assert isinstance(proxy, httpx.Proxy)
    assert str(proxy.url) == "http://10.0.0.5:8080"
    assert proxy.headers["Proxy-Authorization"].startswith("Basic ")
    url, call_kwargs = _RecordingHttpxClient.calls[0]
    assert url == "https://api.elevenlabs.io/v1/voices"
    assert call_kwargs["headers"]["xi-api-key"] == "xi-test-key"


@pytest.mark.asyncio
async def test_elevenlabs_probe_goes_direct_without_a_proxy(monkeypatch):
    _reset_recording()
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingHttpxClient)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="elevenlabs_tts", config={**ELEVENLABS_TTS, "proxy": ""})
    )

    assert result["success"] is True
    assert "directly" in result["message"]
    assert result["proxy"] is None
    assert "proxy" not in _RecordingHttpxClient.kwargs[0]
    # The engine ignores HTTPS_PROXY in the container; so must the probe.
    assert _RecordingHttpxClient.kwargs[0]["trust_env"] is False


@pytest.mark.asyncio
async def test_elevenlabs_probe_rejects_a_proxy_the_engine_would_reject(monkeypatch):
    _reset_recording()
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingHttpxClient)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="elevenlabs_tts", config={**ELEVENLABS_TTS, "proxy": "socks5://10.0.0.5:1080"})
    )

    assert result["success"] is False
    assert "rejected" in result["message"]
    assert "socks5" in result["message"]
    assert _RecordingHttpxClient.calls == []


@pytest.mark.asyncio
async def test_elevenlabs_probe_names_the_proxy_when_the_tunnel_fails(monkeypatch):
    _reset_recording(httpx.ProxyError("403 Forbidden"))
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingHttpxClient)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="elevenlabs_tts", config=dict(ELEVENLABS_TTS))
    )

    assert result["success"] is False
    assert result["message"].startswith("Proxy http://10.0.0.5:8080 refused the tunnel")


@pytest.mark.asyncio
async def test_elevenlabs_probe_names_the_proxy_when_it_is_unreachable(monkeypatch):
    _reset_recording(httpx.ConnectError("Connection refused"))
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingHttpxClient)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="elevenlabs_tts", config=dict(ELEVENLABS_TTS))
    )

    assert result["success"] is False
    assert result["message"].startswith("Cannot connect to proxy http://10.0.0.5:8080")


@pytest.mark.asyncio
async def test_elevenlabs_probe_tells_a_rejected_key_from_a_broken_route(monkeypatch):
    _reset_recording(401)
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingHttpxClient)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="elevenlabs_tts", config=dict(ELEVENLABS_TTS))
    )

    assert result["success"] is False
    assert "Reached ElevenLabs via proxy http://10.0.0.5:8080" in result["message"]
    assert "HTTP 401" in result["message"]


@pytest.mark.asyncio
async def test_openai_probe_of_a_tts_only_block_asks_its_own_host_for_models(monkeypatch):
    """A speech-only openai block has no chat_base_url; its host, not api.openai.com, is probed.

    The probe used to fall back to api.openai.com/v1/models for such a block
    and report that vendor's answer (an HTTP 403 from a blocked region, or a
    401 for the self-hosted token) as the state of the operator's own server.
    """
    _FakeHttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    saved = {
        "type": "openai",
        "capabilities": ["tts"],
        "tts_base_url": "http://10.0.0.5:8091/v1/audio/speech",
        "tts_model": "fishaudio/s2-pro",
    }
    monkeypatch.setattr(
        config_api, "_read_merged_config_dict", lambda: {"providers": {"fish_tts": saved}}
    )

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="fish_tts", config={**saved, "api_key": "local"})
    )

    assert result["success"] is True
    url, kwargs = _FakeHttpxClient.calls[0]
    assert url == "http://10.0.0.5:8091/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer local"


@pytest.mark.asyncio
async def test_openai_probe_of_an_stt_only_block_asks_its_own_host_for_models(monkeypatch):
    _FakeHttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    saved = {
        "type": "openai",
        "capabilities": ["stt"],
        "stt_base_url": "https://stt.example/v1/audio/transcriptions/",
    }
    monkeypatch.setattr(
        config_api, "_read_merged_config_dict", lambda: {"providers": {"whisper_stt": saved}}
    )

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="whisper_stt", config={**saved, "api_key": "self-hosted-token"})
    )

    assert result["success"] is True
    url, _kwargs = _FakeHttpxClient.calls[0]
    assert url == "https://stt.example/v1/models"


@pytest.mark.asyncio
async def test_openai_probe_of_an_unsaved_speech_host_never_falls_back_to_openai(monkeypatch):
    _FakeHttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    monkeypatch.setattr(config_api, "_read_merged_config_dict", lambda: {"providers": {}})

    result = await config_api.test_provider_connection(
        ProviderTestRequest(
            name="fish_tts",
            config={
                "type": "openai",
                "capabilities": ["tts"],
                "tts_base_url": "http://10.0.0.5:8091/v1/audio/speech",
                "api_key": "local",
            },
        )
    )

    assert result["success"] is False
    assert "tts_base_url" in result["message"]
    assert _FakeHttpxClient.calls == []


def test_openai_probe_base_url_drops_only_the_speech_route():
    assert config_api._openai_probe_base_url({"chat_base_url": "https://llm.example/v1"}) == "https://llm.example/v1"
    assert config_api._openai_probe_base_url(
        {"chat_base_url": "https://llm.example/v1", "tts_base_url": "http://tts.lan/v1/audio/speech"}
    ) == "https://llm.example/v1"
    assert config_api._openai_probe_base_url({"tts_base_url": "http://tts.lan:8091/v1/audio/speech"}) == "http://tts.lan:8091/v1"
    assert config_api._openai_probe_base_url({"stt_base_url": "http://stt.lan/v1/audio/transcriptions"}) == "http://stt.lan/v1"
    assert config_api._openai_probe_base_url({"tts_base_url": "http://tts.lan:8091/speech"}) == "http://tts.lan:8091/speech"
    assert config_api._openai_probe_base_url({"tts_base_url": ""}) == ""
    assert config_api._openai_probe_base_url({}) == ""
