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
from fastapi import UploadFile  # noqa: E402

from api.config import ProviderTestRequest  # noqa: E402


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


class _MultipartRelay:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, path, *, fields, sample, timeout=120.0):
        self.calls.append({"path": path, "fields": fields, "sample": sample, "timeout": timeout})
        return self.result


def _upload(name="anna.wav", data=b"RIFF-bytes", content_type="audio/wav"):
    import io

    from starlette.datastructures import Headers

    return UploadFile(file=io.BytesIO(data), filename=name, headers=Headers({"content-type": content_type}))


@pytest.mark.asyncio
async def test_voice_upload_is_relayed_to_the_engine(monkeypatch):
    relay = _MultipartRelay({"success": True, "voice": "anna", "message": "Voice 'anna' registered", "source": "ai_engine"})
    monkeypatch.setattr(engine_relay, "post_engine_multipart", relay)

    result = await config_api.register_provider_voice(
        "fish tts", audio_sample=_upload(), ref_text="Здравствуйте", voice_name=" anna ", consent=None
    )

    assert result["voice"] == "anna"
    call = relay.calls[0]
    assert call["path"] == "/providers/fish%20tts/voices"
    assert call["fields"] == {"ref_text": "Здравствуйте", "name": "anna"}
    assert call["sample"] == ("anna.wav", b"RIFF-bytes", "audio/wav")


@pytest.mark.asyncio
async def test_voice_upload_refuses_empty_and_oversized_samples(monkeypatch):
    relay = _MultipartRelay({"success": True})
    monkeypatch.setattr(engine_relay, "post_engine_multipart", relay)

    with pytest.raises(HTTPException) as empty:
        await config_api.register_provider_voice("fish_tts", audio_sample=_upload(data=b""), ref_text="t")
    with pytest.raises(HTTPException) as huge:
        await config_api.register_provider_voice(
            "fish_tts", audio_sample=_upload(data=b"x" * (10 * 1024 * 1024 + 1)), ref_text="t"
        )

    assert empty.value.status_code == 400
    assert huge.value.status_code == 413
    assert relay.calls == []


@pytest.mark.asyncio
async def test_voice_upload_needs_the_engine(monkeypatch):
    monkeypatch.setattr(engine_relay, "post_engine_multipart", _MultipartRelay(None))

    with pytest.raises(HTTPException) as excinfo:
        await config_api.register_provider_voice("fish_tts", audio_sample=_upload(), ref_text="t")

    assert excinfo.value.status_code == 503
    assert "engine host" in excinfo.value.detail


@pytest.mark.asyncio
async def test_voice_list_is_relayed_to_the_engine(monkeypatch):
    seen = {}

    async def fake_get(path, *, timeout=30.0):
        seen["path"] = path
        return {"success": True, "voices": [{"name": "anna"}], "builtin": ["default"], "message": "1 registered voice(s)"}

    monkeypatch.setattr(engine_relay, "get_engine_json", fake_get)

    result = await config_api.list_provider_voices("fish_tts")

    assert seen["path"] == "/providers/fish_tts/voices"
    assert result["voices"] == [{"name": "anna"}]

    async def unreachable(path, *, timeout=30.0):
        return None

    monkeypatch.setattr(engine_relay, "get_engine_json", unreachable)
    with pytest.raises(HTTPException) as excinfo:
        await config_api.list_provider_voices("fish_tts")
    assert excinfo.value.status_code == 503


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

    async def request(self, method, url, headers=None, **kwargs):
        import httpx

        _HttpxClient.calls.append({"method": method, "url": url, "headers": headers, **kwargs})
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

    def fake_exec(method, path, payload, timeout):
        seen.update(method=method, path=path, payload=payload)
        return {"success": False, "message": "from exec"}

    monkeypatch.setattr(engine_relay, "_exec_request_sync", fake_exec)

    result = await engine_relay.post_engine_json("/providers/test", {"name": "x", "config": {}})

    assert result == {"success": False, "message": "from exec"}
    assert seen == {"method": "POST", "path": "/providers/test", "payload": {"name": "x", "config": {}}}


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


@pytest.mark.asyncio
async def test_multipart_relay_sends_the_file_over_the_network(monkeypatch):
    import httpx

    _HttpxClient.answers = {"http://ai_engine:15000": _Response(200, {"success": True, "voice": "anna"})}
    _HttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _HttpxClient)
    monkeypatch.setattr(engine_relay, "_engine_base_urls", lambda: ["http://ai_engine:15000"])
    monkeypatch.setattr(engine_relay, "_health_api_token", lambda: "tok-1")

    result = await engine_relay.post_engine_multipart(
        "/providers/fish_tts/voices", fields={"ref_text": "t", "name": "anna"}, sample=("anna.wav", b"RIFF", "audio/wav")
    )

    assert result == {"success": True, "voice": "anna"}
    call = _HttpxClient.calls[0]
    assert call["method"] == "POST"
    assert call["data"] == {"ref_text": "t", "name": "anna"}
    assert call["files"] == {"audio_sample": ("anna.wav", b"RIFF", "audio/wav")}
    assert call["headers"] == {"Authorization": "Bearer tok-1"}


@pytest.mark.asyncio
async def test_multipart_relay_stages_the_file_through_the_docker_socket(monkeypatch):
    import httpx

    _HttpxClient.answers = {}
    _HttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _HttpxClient)
    monkeypatch.setattr(engine_relay, "_engine_base_urls", lambda: ["http://127.0.0.1:15000"])
    monkeypatch.setattr(engine_relay, "_health_api_token", lambda: "")
    staged = {}

    def fake_stage(filename, data):
        staged.update(filename=filename, data=data)
        return "deadbeef.wav"

    seen = {}

    def fake_exec(method, path, payload, timeout):
        seen.update(method=method, path=path, payload=payload)
        return {"success": True, "voice": "anna"}

    monkeypatch.setattr(engine_relay, "_stage_sample_sync", fake_stage)
    monkeypatch.setattr(engine_relay, "_exec_request_sync", fake_exec)

    result = await engine_relay.post_engine_multipart(
        "/providers/fish_tts/voices", fields={"ref_text": "t"}, sample=("anna.wav", b"RIFF", "audio/wav")
    )

    assert result == {"success": True, "voice": "anna"}
    assert staged == {"filename": "anna.wav", "data": b"RIFF"}
    assert seen == {
        "method": "POST",
        "path": "/providers/fish_tts/voices",
        "payload": {"ref_text": "t", "staged_file": "deadbeef.wav", "filename": "anna.wav"},
    }


def test_staging_archive_holds_the_sample_under_the_staging_directory(monkeypatch):
    import io
    import tarfile

    class _Container:
        def __init__(self):
            self.archives = []

        def put_archive(self, path, data):
            self.archives.append((path, data))
            return True

    container = _Container()
    monkeypatch.setattr(engine_relay, "_engine_container", lambda: container)

    staged = engine_relay._stage_sample_sync("Anna Voice.MP3", b"ID3-bytes")

    assert staged.endswith(".mp3") and "/" not in staged
    path, data = container.archives[0]
    assert path == "/tmp"
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        names = archive.getnames()
        member = archive.extractfile(f"ava-voice-uploads/{staged}")
        assert member is not None and member.read() == b"ID3-bytes"
    assert names == ["ava-voice-uploads", f"ava-voice-uploads/{staged}"]


@pytest.mark.asyncio
async def test_get_relay_carries_no_body(monkeypatch):
    import httpx

    _HttpxClient.answers = {"http://127.0.0.1:15000": _Response(200, {"success": True, "voices": []})}
    _HttpxClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _HttpxClient)
    monkeypatch.setattr(engine_relay, "_engine_base_urls", lambda: ["http://127.0.0.1:15000"])
    monkeypatch.setattr(engine_relay, "_health_api_token", lambda: "")

    result = await engine_relay.get_engine_json("/providers/fish_tts/voices")

    assert result == {"success": True, "voices": []}
    assert _HttpxClient.calls[0]["method"] == "GET"
    assert "json" not in _HttpxClient.calls[0]
