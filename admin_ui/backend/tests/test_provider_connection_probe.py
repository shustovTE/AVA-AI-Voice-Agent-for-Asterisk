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
