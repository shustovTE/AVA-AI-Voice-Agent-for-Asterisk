"""The engine's health server answers provider tests and voice registrations for the Admin UI."""
import json
from types import SimpleNamespace

import pytest

from src.engine import Engine
from src.probes import providers, voices


class _Request:
    def __init__(self, payload=None, *, match_info=None, invalid=False):
        self._payload = payload
        self._invalid = invalid
        self.match_info = match_info or {}

    async def json(self):
        if self._invalid:
            raise ValueError("not json")
        return self._payload


def _engine(saved=None, *, authorized=True):
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(providers={})
    engine._is_request_authorized = lambda _request: authorized
    engine._saved_provider_blocks = lambda: dict(saved or {})
    return engine


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
async def test_voice_registration_uses_the_saved_block_not_the_request(monkeypatch):
    seen = {}

    async def fake_register(provider_key, block, *, file, name=None, ref_text, consent=None, env=None):
        seen.update(key=provider_key, block=block, file=file, name=name, ref_text=ref_text, consent=consent)
        return {"success": True, "voice": name or "anna", "message": "ok"}

    monkeypatch.setattr(voices, "register_voice", fake_register)
    saved = {"Fish_TTS": {"type": "openai", "tts_base_url": "http://127.0.0.1:8091/v1/audio/speech"}}
    request = _Request(
        {"file": "anna.wav", "name": "anna", "ref_text": "Привет", "consent": "c-1", "tts_base_url": "http://evil/"},
        match_info={"name": "fish_tts"},
    )

    response = await Engine._provider_voice_register_handler(_engine(saved), request)

    assert response.status == 200
    assert json.loads(response.text)["source"] == "ai_engine"
    assert seen["block"] == saved["Fish_TTS"]
    assert seen["file"] == "anna.wav" and seen["name"] == "anna" and seen["ref_text"] == "Привет" and seen["consent"] == "c-1"
