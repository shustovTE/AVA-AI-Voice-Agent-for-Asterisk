"""Provider "Test connection" runs in ai_engine, where the calls run.

The Admin UI container may sit in another Docker network than the engine
host's endpoints (a vLLM on the host's loopback, an OpenAI-compatible LLM on
the host, an ElevenLabs proxy only the host reaches), so a probe from here
reported them dead. The backend now relays to the engine and runs the same
probe itself only when the engine cannot be reached, saying so in the verdict.
"""
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from api import config as config_api  # noqa: E402
from api import engine_relay  # noqa: E402
from api.config import ProviderTestRequest, VoiceRegisterRequest  # noqa: E402


class _Relay:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    async def __call__(self, path, payload, *, timeout=30.0):
        self.calls.append({"path": path, "payload": payload, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.mark.asyncio
async def test_probe_is_relayed_to_the_engine(monkeypatch):
    relay = _Relay({"success": True, "message": "Connected (OpenAI-compatible). Found 3 models.", "source": "ai_engine"})
    monkeypatch.setattr(engine_relay, "post_engine_json", relay)

    async def never(*_args, **_kwargs):
        raise AssertionError("the local probe must not run while the engine answers")

    monkeypatch.setattr(config_api, "probe_provider", never)
    config = {"type": "openai", "tts_base_url": "http://127.0.0.1:8091/v1/audio/speech", "api_key": "local"}

    result = await config_api.test_provider_connection(ProviderTestRequest(name="fish_tts", config=config))

    assert relay.calls == [{"path": "/providers/test", "payload": {"name": "fish_tts", "config": config}, "timeout": 60.0}]
    assert result["success"] is True
    assert result["source"] == "ai_engine"
    assert result["message"] == "Connected (OpenAI-compatible). Found 3 models. (checked from ai_engine)"


@pytest.mark.asyncio
async def test_probe_runs_locally_only_when_the_engine_is_unreachable(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_relay, "post_engine_json", _Relay(None))
    env_file = tmp_path / ".env"
    env_file.write_text('OPENAI_API_KEY="sk-from-dotenv"\n# comment\nEMPTY=\n')
    monkeypatch.setattr(config_api.settings, "ENV_PATH", str(env_file))
    saved = {"native_llm": {"type": "openai", "chat_base_url": "https://ai-api.example/v1"}}
    monkeypatch.setattr(config_api, "_read_merged_config_dict", lambda: {"providers": saved})
    seen = {}

    async def fake_probe(name, config, *, saved_providers=None, env=None):
        seen.update(name=name, config=config, saved=saved_providers, env=env)
        return {"success": False, "message": "Cannot connect to provider at https://ai-api.example/v1 (see server logs)"}

    monkeypatch.setattr(config_api, "probe_provider", fake_probe)

    result = await config_api.test_provider_connection(
        ProviderTestRequest(name="native_llm", config={"type": "openai", "chat_base_url": "https://ai-api.example/v1"})
    )

    assert seen["name"] == "native_llm"
    assert seen["saved"] == saved
    assert seen["env"] == {"OPENAI_API_KEY": "sk-from-dotenv", "EMPTY": ""}
    assert result["success"] is False
    assert result["source"] == "admin_ui"
    assert result["message"].endswith(
        "(ai_engine unreachable, checked from the Admin UI container, "
        "which may not reach endpoints only the engine host sees)"
    )


@pytest.mark.asyncio
async def test_an_engine_error_reaches_the_browser_as_is(monkeypatch):
    monkeypatch.setattr(engine_relay, "post_engine_json", _Relay(exc=HTTPException(status_code=504, detail="AI Engine did not answer in time")))

    with pytest.raises(HTTPException) as excinfo:
        await config_api.test_provider_connection(ProviderTestRequest(name="x", config={"type": "openai"}))

    assert excinfo.value.status_code == 504


@pytest.mark.asyncio
async def test_voice_registration_is_relayed_to_the_engine(monkeypatch):
    relay = _Relay({"success": True, "voice": "anna", "message": "Voice 'anna' registered", "source": "ai_engine"})
    monkeypatch.setattr(engine_relay, "post_engine_json", relay)

    result = await config_api.register_provider_voice(
        "fish tts", VoiceRegisterRequest(file="anna.wav", ref_text="Здравствуйте", consent=None)
    )

    assert result["voice"] == "anna"
    assert relay.calls[0]["path"] == "/providers/fish%20tts/voices"
    assert relay.calls[0]["payload"] == {"file": "anna.wav", "name": None, "ref_text": "Здравствуйте", "consent": None}


@pytest.mark.asyncio
async def test_voice_registration_needs_the_engine(monkeypatch):
    monkeypatch.setattr(engine_relay, "post_engine_json", _Relay(None))

    with pytest.raises(HTTPException) as excinfo:
        await config_api.register_provider_voice("fish_tts", VoiceRegisterRequest(file="anna.wav", ref_text="t"))

    assert excinfo.value.status_code == 503
    assert "engine host" in excinfo.value.detail


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _HttpxClient:
    """Scripts one answer per base URL; a missing base is a connection error."""

    answers = {}
    calls = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, headers=None, json=None):
        import httpx

        _HttpxClient.calls.append({"url": url, "headers": headers, "json": json})
        for base, answer in _HttpxClient.answers.items():
            if url.startswith(base):
                return answer
        raise httpx.ConnectError("refused")


@pytest.mark.asyncio
async def test_relay_tries_each_engine_address_with_the_token(monkeypatch):
    import httpx

    _HttpxClient.answers = {"http://ai_engine:15000": _Response(200, {"success": True, "message": "ok"})}
    _HttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _HttpxClient)
    monkeypatch.setattr(engine_relay, "_engine_base_urls", lambda: ["http://127.0.0.1:15000", "http://ai_engine:15000"])
    monkeypatch.setattr(engine_relay, "_health_api_token", lambda: "tok-1")

    result = await engine_relay.post_engine_json("/providers/test", {"name": "x", "config": {}})

    assert result == {"success": True, "message": "ok"}
    assert [c["url"] for c in _HttpxClient.calls] == [
        "http://127.0.0.1:15000/providers/test",
        "http://ai_engine:15000/providers/test",
    ]
    assert _HttpxClient.calls[1]["headers"] == {"Authorization": "Bearer tok-1"}


@pytest.mark.asyncio
async def test_relay_falls_back_to_the_docker_socket_when_no_address_answers(monkeypatch):
    import httpx

    _HttpxClient.answers = {}
    _HttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _HttpxClient)
    monkeypatch.setattr(engine_relay, "_engine_base_urls", lambda: ["http://127.0.0.1:15000"])
    monkeypatch.setattr(engine_relay, "_health_api_token", lambda: "")
    seen = {}

    def fake_exec(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"success": False, "message": "from exec"}

    monkeypatch.setattr(engine_relay, "_exec_post_sync", fake_exec)

    result = await engine_relay.post_engine_json("/providers/test", {"name": "x", "config": {}})

    assert result == {"success": False, "message": "from exec"}
    assert seen == {"path": "/providers/test", "payload": {"name": "x", "config": {}}}


@pytest.mark.asyncio
async def test_relay_surfaces_an_engine_refusal(monkeypatch):
    import httpx

    _HttpxClient.answers = {"http://127.0.0.1:15000": _Response(404, {"success": False, "message": "Provider 'x' is not saved"})}
    _HttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _HttpxClient)
    monkeypatch.setattr(engine_relay, "_engine_base_urls", lambda: ["http://127.0.0.1:15000"])
    monkeypatch.setattr(engine_relay, "_health_api_token", lambda: "")

    # A verdict (it carries "success") is returned as data, whatever the status.
    result = await engine_relay.post_engine_json("/providers/x/voices", {"file": "a.wav"})
    assert result == {"success": False, "message": "Provider 'x' is not saved"}

    _HttpxClient.answers = {"http://127.0.0.1:15000": _Response(500, {"error": "internal_error"})}
    with pytest.raises(HTTPException) as excinfo:
        await engine_relay.post_engine_json("/providers/test", {"name": "x", "config": {}})
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "internal_error"
