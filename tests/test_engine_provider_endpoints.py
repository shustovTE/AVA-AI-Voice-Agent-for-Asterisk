"""The engine's health server answers provider tests, voice registrations and voice lists for the Admin UI."""
import io
import json
from types import SimpleNamespace

import pytest

from src.engine import Engine
from src.probes import providers, voices


class _Request:
    def __init__(self, payload=None, *, match_info=None, invalid=False, form=None, content_type="application/json"):
        self._payload = payload
        self._invalid = invalid
        self._form = form
        self.match_info = match_info or {}
        self.content_type = content_type

    async def json(self):
        if self._invalid:
            raise ValueError("not json")
        return self._payload

    async def post(self):
        return self._form


def _engine(saved=None, *, authorized=True):
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(providers={})
    engine._is_request_authorized = lambda _request: authorized
    engine._saved_provider_blocks = lambda: dict(saved or {})
    return engine


SAVED = {"Fish_TTS": {"type": "openai", "tts_base_url": "http://127.0.0.1:8091/v1/audio/speech"}}


def _fake_register(seen):
    async def fake(provider_key, block, *, sample, filename, name=None, ref_text, consent=None, env=None):
        seen.update(key=provider_key, block=block, sample=sample, filename=filename, name=name, ref_text=ref_text, consent=consent)
        return {"success": True, "voice": name or "anna", "message": "ok"}

    return fake


@pytest.mark.asyncio
async def test_provider_test_requires_authorization():
    response = await Engine._provider_test_handler(_engine(authorized=False), _Request({"name": "x", "config": {}}))

    assert response.status == 403


@pytest.mark.asyncio
async def test_provider_test_rejects_a_malformed_body():
    engine = _engine()

    not_json = await Engine._provider_test_handler(engine, _Request(invalid=True))
    no_config = await Engine._provider_test_handler(engine, _Request({"name": "fish_tts"}))

    assert not_json.status == 400
    assert no_config.status == 400


@pytest.mark.asyncio
async def test_provider_test_runs_the_probe_against_the_saved_blocks(monkeypatch):
    seen = {}

    async def fake_probe(name, config, *, saved_providers=None, env=None):
        seen.update(name=name, config=config, saved=saved_providers)
        return {"success": True, "message": "Connected"}

    monkeypatch.setattr(providers, "probe_provider", fake_probe)
    saved = {"fish_tts": {"type": "openai", "tts_base_url": "http://127.0.0.1:8091/v1/audio/speech"}}
    request = _Request({"name": "fish_tts", "config": {"type": "openai", "api_key": "local"}})

    response = await Engine._provider_test_handler(_engine(saved), request)

    assert response.status == 200
    assert json.loads(response.text) == {"success": True, "message": "Connected", "source": "ai_engine"}
    assert seen == {"name": "fish_tts", "config": {"type": "openai", "api_key": "local"}, "saved": saved}


@pytest.mark.asyncio
async def test_voice_registration_needs_a_saved_provider():
    request = _Request({"file": "anna.wav", "ref_text": "t"}, match_info={"name": "fish_tts"})

    response = await Engine._provider_voice_register_handler(_engine({}), request)

    assert response.status == 404
    assert "not saved" in json.loads(response.text)["message"]


@pytest.mark.asyncio
async def test_voice_registration_takes_a_multipart_upload(monkeypatch):
    seen = {}
    monkeypatch.setattr(voices, "register_voice_bytes", _fake_register(seen))
    form = {
        "audio_sample": SimpleNamespace(file=io.BytesIO(b"RIFF-bytes"), filename="anna.wav"),
        "name": "anna",
        "ref_text": "Привет",
        "consent": "c-1",
    }
    request = _Request(match_info={"name": "fish_tts"}, form=form, content_type="multipart/form-data; boundary=xyz")

    response = await Engine._provider_voice_register_handler(_engine(SAVED), request)

    assert response.status == 200
    assert json.loads(response.text)["source"] == "ai_engine"
    assert seen["block"] == SAVED["Fish_TTS"]
    assert seen["sample"] == b"RIFF-bytes" and seen["filename"] == "anna.wav"
    assert seen["name"] == "anna" and seen["ref_text"] == "Привет" and seen["consent"] == "c-1"


@pytest.mark.asyncio
async def test_voice_registration_without_a_file_part_is_refused():
    form = {"ref_text": "Привет"}
    request = _Request(match_info={"name": "fish_tts"}, form=form, content_type="multipart/form-data; boundary=xyz")

    response = await Engine._provider_voice_register_handler(_engine(SAVED), request)

    assert response.status == 400
    assert "audio_sample" in json.loads(response.text)["message"]


@pytest.mark.asyncio
async def test_voice_registration_takes_a_staged_file_and_removes_it(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(voices, "register_voice_bytes", _fake_register(seen))
    monkeypatch.setattr(voices, "STAGING_DIR", tmp_path)
    (tmp_path / "deadbeef.wav").write_bytes(b"staged-bytes")
    request = _Request(
        {"staged_file": "deadbeef.wav", "filename": "anna.wav", "ref_text": "Привет"},
        match_info={"name": "fish_tts"},
    )

    response = await Engine._provider_voice_register_handler(_engine(SAVED), request)

    assert response.status == 200
    assert seen["sample"] == b"staged-bytes" and seen["filename"] == "anna.wav"
    assert not (tmp_path / "deadbeef.wav").exists()

    gone = await Engine._provider_voice_register_handler(_engine(SAVED), request)
    assert gone.status == 400
    assert "not found" in json.loads(gone.text)["message"]


@pytest.mark.asyncio
async def test_voice_registration_takes_a_file_of_the_voices_directory(monkeypatch):
    seen = {}

    async def fake_register(provider_key, block, *, file, name=None, ref_text, consent=None, env=None):
        seen.update(key=provider_key, block=block, file=file, ref_text=ref_text)
        return {"success": True, "voice": "anna", "message": "ok"}

    monkeypatch.setattr(voices, "register_voice", fake_register)
    request = _Request({"file": "anna.wav", "ref_text": "Привет", "tts_base_url": "http://evil/"}, match_info={"name": "fish_tts"})

    response = await Engine._provider_voice_register_handler(_engine(SAVED), request)

    assert response.status == 200
    assert seen == {"key": "fish_tts", "block": SAVED["Fish_TTS"], "file": "anna.wav", "ref_text": "Привет"}


@pytest.mark.asyncio
async def test_voice_registration_json_without_a_source_is_refused():
    request = _Request({"ref_text": "Привет"}, match_info={"name": "fish_tts"})

    response = await Engine._provider_voice_register_handler(_engine(SAVED), request)

    assert response.status == 400


@pytest.mark.asyncio
async def test_voice_list_uses_the_saved_block(monkeypatch):
    seen = {}

    async def fake_list(provider_key, block, *, env=None):
        seen.update(key=provider_key, block=block)
        return {"success": True, "voices": [{"name": "anna"}], "builtin": ["default"], "message": "1 registered voice(s)"}

    monkeypatch.setattr(voices, "list_voices", fake_list)

    unknown = await Engine._provider_voices_list_handler(_engine({}), _Request(match_info={"name": "fish_tts"}))
    response = await Engine._provider_voices_list_handler(_engine(SAVED), _Request(match_info={"name": "fish_tts"}))
    forbidden = await Engine._provider_voices_list_handler(_engine(SAVED, authorized=False), _Request(match_info={"name": "fish_tts"}))

    assert unknown.status == 404
    assert forbidden.status == 403
    assert response.status == 200
    assert json.loads(response.text)["voices"] == [{"name": "anna"}]
    assert seen == {"key": "fish_tts", "block": SAVED["Fish_TTS"]}
