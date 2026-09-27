"""The provider probes run in the engine, where the calls run.

They used to live in the Admin UI backend, in another Docker network than
the engine host's endpoints, and reported a self-hosted vLLM or a proxied
ElevenLabs leg dead. The cases here pin the verdicts and the routes taken.
"""
import json
from typing import Any, Dict, List

import aiohttp
import pytest
import websockets

from src.probes import providers
from src.probes.providers import HttpResponse, openai_probe_base_url, probe_provider, substitute_env_vars


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
    return json.dumps(
        {
            "type": "status_response",
            "status": "ok",
            "stt_backend": "tone",
            "tts_backend": "piper",
            "models": {"stt": {"loaded": stt}, "llm": {"loaded": llm, "path": ""}, "tts": {"loaded": tts}},
            "config": {"runtime_mode": runtime_mode},
        }
    )


AUTH_OK = '{"type": "auth_response", "status": "ok"}'
AUTH_FAIL = '{"type": "auth_response", "status": "error", "message": "invalid_auth_token"}'


class _Http:
    """Records every request the probe makes and answers as scripted."""

    def __init__(self, status=200, body: Any = None, exc: Exception | None = None):
        self.calls: List[Dict[str, Any]] = []
        self.status = status
        self.body = body if body is not None else {"data": [{"id": "m"}]}
        self.exc = exc

    async def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        text = self.body if isinstance(self.body, str) else json.dumps(self.body)
        return HttpResponse(self.status, text)


@pytest.mark.asyncio
async def test_local_probe_authenticates_and_scopes_to_declared_capabilities(monkeypatch):
    seen = []
    monkeypatch.setattr(websockets, "connect", _fake_connect([AUTH_OK, _status_response()], seen))

    result = await probe_provider(
        "local_stt",
        {"type": "local", "capabilities": ["stt"], "ws_url": "ws://172.17.0.1:8765", "auth_token": "secret-token"},
    )

    assert seen == ["ws://172.17.0.1:8765"]
    assert result["success"] is True
    assert "STT: tone" in result["message"]


@pytest.mark.asyncio
async def test_local_probe_reports_rejected_token(monkeypatch):
    monkeypatch.setattr(websockets, "connect", _fake_connect([AUTH_FAIL], []))

    result = await probe_provider(
        "local_stt",
        {"type": "local", "capabilities": ["stt"], "ws_url": "ws://172.17.0.1:8765", "auth_token": "wrong"},
    )

    assert result["success"] is False
    assert "auth token" in result["message"].lower()


@pytest.mark.asyncio
async def test_local_probe_does_not_require_llm_in_minimal_mode(monkeypatch):
    monkeypatch.setattr(
        websockets,
        "connect",
        _fake_connect([_status_response(stt=True, llm=False, tts=True, runtime_mode="minimal")], []),
    )

    result = await probe_provider("local", {"type": "local", "ws_url": "ws://127.0.0.1:8765"})

    assert result["success"] is True


@pytest.mark.asyncio
async def test_openai_probe_calls_saved_self_hosted_endpoint(monkeypatch):
    http = _Http()
    monkeypatch.setattr(providers, "_http_request", http)
    saved = {"native_llm": {"type": "openai", "chat_base_url": "https://ai-api.example/v1"}}

    result = await probe_provider(
        "native_llm",
        {"type": "openai", "chat_base_url": "https://ai-api.example/v1", "api_key": "self-hosted-token"},
        saved_providers=saved,
    )

    assert result["success"] is True
    assert http.calls[0]["url"] == "https://ai-api.example/v1/models"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer self-hosted-token"


@pytest.mark.asyncio
async def test_openai_probe_never_sends_the_key_to_an_unsaved_host(monkeypatch):
    http = _Http()
    monkeypatch.setattr(providers, "_http_request", http)
    saved = {"native_llm": {"type": "openai", "chat_base_url": "https://ai-api.example/v1"}}

    result = await probe_provider(
        "native_llm",
        {"type": "openai", "chat_base_url": "https://attacker.example/v1", "api_key": "self-hosted-token"},
        saved_providers=saved,
    )

    assert result["success"] is False
    assert http.calls == []


@pytest.mark.asyncio
async def test_openai_probe_of_a_tts_only_block_asks_its_own_host_for_models(monkeypatch):
    http = _Http()
    monkeypatch.setattr(providers, "_http_request", http)
    saved = {"fish_tts": {"type": "openai", "tts_base_url": "http://127.0.0.1:8091/v1/audio/speech"}}

    result = await probe_provider(
        "fish_tts",
        {"type": "openai", "capabilities": ["tts"], "tts_base_url": "http://127.0.0.1:8091/v1/audio/speech", "api_key": "local"},
        saved_providers=saved,
    )

    assert result["success"] is True
    assert http.calls[0]["url"] == "http://127.0.0.1:8091/v1/models"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer local"


@pytest.mark.asyncio
async def test_openai_probe_reports_the_vendor_status(monkeypatch):
    monkeypatch.setattr(providers, "_http_request", _Http(status=401))

    result = await probe_provider("openai_llm", {"type": "openai", "api_key": "sk-bad"})

    assert result == {"success": False, "message": "Invalid API key (401)"}


def test_openai_probe_base_url_drops_only_the_speech_route():
    assert openai_probe_base_url({"chat_base_url": "https://llm.example/v1"}) == "https://llm.example/v1"
    assert openai_probe_base_url(
        {"chat_base_url": "https://llm.example/v1", "tts_base_url": "http://tts.lan/v1/audio/speech"}
    ) == "https://llm.example/v1"
    assert openai_probe_base_url({"tts_base_url": "http://tts.lan:8091/v1/audio/speech"}) == "http://tts.lan:8091/v1"
    assert openai_probe_base_url({"stt_base_url": "http://stt.lan/v1/audio/transcriptions/"}) == "http://stt.lan/v1"
    assert openai_probe_base_url({"tts_base_url": "http://tts.lan:8091/speech"}) == "http://tts.lan:8091/speech"
    assert openai_probe_base_url({"tts_base_url": ""}) == ""
    assert openai_probe_base_url({}) == ""


def test_env_placeholders_take_the_overlay_then_the_default():
    env = {"LLM_URL": "http://10.0.0.5:8000/v1"}
    block = {"chat_base_url": "${LLM_URL:-http://fallback/v1}", "chat_model": "${LLM_MODEL:-qwen}", "nested": ["${X}"]}
    assert substitute_env_vars(block, env) == {
        "chat_base_url": "http://10.0.0.5:8000/v1",
        "chat_model": "qwen",
        "nested": [""],
    }


ELEVENLABS_TTS = {
    "type": "elevenlabs",
    "capabilities": ["tts"],
    "api_key": "xi-test-key",
    "proxy": "http://user:secret@10.0.0.5:8080",
}


@pytest.mark.asyncio
async def test_elevenlabs_probe_takes_the_configured_proxy(monkeypatch):
    """A modular TTS provider of type elevenlabs is probed through its proxy, as the adapter goes."""
    http = _Http(body={"voices": [{"id": "a"}, {"id": "b"}]})
    monkeypatch.setattr(providers, "_http_request", http)

    result = await probe_provider("eleven_tts", dict(ELEVENLABS_TTS))

    assert result["success"] is True
    assert "via proxy http://10.0.0.5:8080" in result["message"]
    assert "secret" not in result["message"]
    assert result["proxy"] == "http://10.0.0.5:8080"
    call = http.calls[0]
    assert call["url"] == "https://api.elevenlabs.io/v1/voices"
    assert call["headers"]["xi-api-key"] == "xi-test-key"
    assert call["proxy"] == "http://10.0.0.5:8080"
    assert call["proxy_headers"]["Proxy-Authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_elevenlabs_probe_goes_direct_without_a_proxy(monkeypatch):
    http = _Http(body={"voices": []})
    monkeypatch.setattr(providers, "_http_request", http)

    result = await probe_provider("elevenlabs_tts", {**ELEVENLABS_TTS, "proxy": ""})

    assert result["success"] is True
    assert "directly" in result["message"]
    assert result["proxy"] is None
    assert http.calls[0]["proxy"] is None


@pytest.mark.asyncio
async def test_elevenlabs_probe_rejects_a_proxy_the_engine_would_reject(monkeypatch):
    http = _Http()
    monkeypatch.setattr(providers, "_http_request", http)

    result = await probe_provider("elevenlabs_tts", {**ELEVENLABS_TTS, "proxy": "socks5://10.0.0.5:1080"})

    assert result["success"] is False
    assert "rejected" in result["message"]
    assert "socks5" in result["message"]
    assert http.calls == []


class _ProxyDown(aiohttp.ClientProxyConnectionError):
    def __init__(self):
        Exception.__init__(self, "Connection refused")

    def __str__(self):
        return "Connection refused"


class _TunnelRefused(aiohttp.ClientHttpProxyError):
    def __init__(self):
        Exception.__init__(self, "403 Forbidden")

    def __str__(self):
        return "403 Forbidden"


@pytest.mark.asyncio
async def test_elevenlabs_probe_names_the_proxy_when_it_is_unreachable(monkeypatch):
    monkeypatch.setattr(providers, "_http_request", _Http(exc=_ProxyDown()))

    result = await probe_provider("elevenlabs_tts", dict(ELEVENLABS_TTS))

    assert result["success"] is False
    assert result["message"].startswith("Cannot connect to proxy http://10.0.0.5:8080")


@pytest.mark.asyncio
async def test_elevenlabs_probe_names_the_proxy_when_the_tunnel_fails(monkeypatch):
    monkeypatch.setattr(providers, "_http_request", _Http(exc=_TunnelRefused()))

    result = await probe_provider("elevenlabs_tts", dict(ELEVENLABS_TTS))

    assert result["success"] is False
    assert result["message"].startswith("Proxy http://10.0.0.5:8080 refused the tunnel")


@pytest.mark.asyncio
async def test_elevenlabs_probe_tells_a_rejected_key_from_a_broken_route(monkeypatch):
    monkeypatch.setattr(providers, "_http_request", _Http(status=401, body={}))

    result = await probe_provider("elevenlabs_tts", dict(ELEVENLABS_TTS))

    assert result["success"] is False
    assert "Reached ElevenLabs via proxy http://10.0.0.5:8080" in result["message"]
    assert "HTTP 401" in result["message"]


@pytest.mark.asyncio
async def test_groq_probe_is_pinned_to_the_vendor_endpoint(monkeypatch):
    http = _Http()
    monkeypatch.setattr(providers, "_http_request", http)

    result = await probe_provider(
        "groq_tts", {"type": "groq", "api_key": "gsk-test", "tts_base_url": "http://evil.example/v1"}
    )

    assert result["success"] is True
    assert http.calls[0]["url"] == "https://api.groq.com/openai/v1/models"


@pytest.mark.asyncio
async def test_unknown_block_is_reported_not_guessed():
    result = await probe_provider("mystery", {"type": "unknown-vendor"})

    assert result == {"success": False, "message": "Unknown provider type - cannot test"}


@pytest.mark.asyncio
async def test_a_crashing_probe_still_returns_a_verdict(monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("socket exploded")

    monkeypatch.setattr(providers, "_http_request", boom)

    result = await probe_provider("openai_realtime", {"realtime_base_url": "wss://x", "api_key": "k"}, env={"OPENAI_API_KEY": "sk"})

    assert result["success"] is False
    assert "socket exploded" in result["message"]
