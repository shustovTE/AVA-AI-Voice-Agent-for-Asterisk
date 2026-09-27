"""Registering a reference voice on a self-hosted speech endpoint from the engine host."""
import json

import aiohttp
import pytest

from src.probes import voices
from src.probes.voices import register_voice, resolve_sample_path


BLOCK = {
    "type": "openai",
    "api_key": "local",
    "tts_base_url": "http://tts.lan:8091/v1/audio/speech",
    "tts_model": "fishaudio/s2-pro",
}


class _Server:
    def __init__(self, status=200, body=None, exc=None):
        self.calls = []
        self.status = status
        self.body = body if body is not None else {"success": True, "voice": {"name": "anna", "created_at": 1}}
        self.exc = exc

    async def __call__(self, url, form, headers, timeout):
        fields = {}
        for options, _headers, value in form._fields:
            fields[options["name"]] = value
        self.calls.append({"url": url, "fields": fields, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.status, json.dumps(self.body)


@pytest.fixture
def voices_dir(tmp_path):
    (tmp_path / "anna.wav").write_bytes(b"RIFF" + b"\x00" * 2000)
    return tmp_path


@pytest.mark.asyncio
async def test_register_voice_uploads_the_sample_with_its_transcript(monkeypatch, voices_dir):
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)
    block = {**BLOCK, "tts_voices_dir": str(voices_dir)}

    result = await register_voice("fish_tts", block, file="anna.wav", ref_text="Здравствуйте, это Анна.")

    assert result["success"] is True
    assert result["voice"] == "anna"
    assert "registered at tts.lan" in result["message"]
    call = server.calls[0]
    assert call["url"] == "http://tts.lan:8091/v1/audio/voices"
    assert call["headers"] == {"Authorization": "Bearer local"}
    assert call["fields"]["name"] == "anna"
    assert call["fields"]["ref_text"] == "Здравствуйте, это Анна."
    assert call["fields"]["consent"] == "ava-admin"
    assert call["fields"]["audio_sample"].startswith(b"RIFF")


@pytest.mark.asyncio
async def test_register_voice_takes_an_explicit_name_and_consent(monkeypatch, voices_dir):
    server = _Server(body={"success": True, "voice": {"name": "Anna-RU"}})
    monkeypatch.setattr(voices, "_post_form", server)
    block = {**BLOCK, "tts_voices_dir": str(voices_dir)}

    result = await register_voice(
        "fish_tts", block, file="anna.wav", name="Anna-RU", ref_text="text", consent="consent-2026"
    )

    assert result["voice"] == "Anna-RU"
    assert server.calls[0]["fields"]["name"] == "Anna-RU"
    assert server.calls[0]["fields"]["consent"] == "consent-2026"


@pytest.mark.asyncio
async def test_register_voice_relays_the_servers_refusal(monkeypatch, voices_dir):
    server = _Server(status=400, body={"error": {"message": "Reference audio too short (0.5s).", "type": "BadRequestError"}})
    monkeypatch.setattr(voices, "_post_form", server)
    block = {**BLOCK, "tts_voices_dir": str(voices_dir)}

    result = await register_voice("fish_tts", block, file="anna.wav", ref_text="text")

    assert result["success"] is False
    assert "HTTP 400: Reference audio too short (0.5s)." in result["message"]


class _Unreachable(aiohttp.ClientConnectorError):
    def __init__(self):
        Exception.__init__(self, "refused")

    def __str__(self):
        return "Connection refused"


@pytest.mark.asyncio
async def test_register_voice_names_an_unreachable_server(monkeypatch, voices_dir):
    monkeypatch.setattr(voices, "_post_form", _Server(exc=_Unreachable()))
    block = {**BLOCK, "tts_voices_dir": str(voices_dir)}

    result = await register_voice("fish_tts", block, file="anna.wav", ref_text="text")

    assert result["success"] is False
    assert result["message"].startswith("Cannot connect to the speech server at http://tts.lan:8091/v1")


@pytest.mark.asyncio
async def test_register_voice_refuses_paths_missing_files_and_empty_transcripts(monkeypatch, voices_dir):
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)
    block = {**BLOCK, "tts_voices_dir": str(voices_dir)}

    traversal = await register_voice("fish_tts", block, file="../etc/passwd", ref_text="text")
    missing = await register_voice("fish_tts", block, file="nobody.wav", ref_text="text")
    silent = await register_voice("fish_tts", block, file="anna.wav", ref_text="   ")

    assert traversal["success"] is False and "path separators" in traversal["message"]
    assert missing["success"] is False and "not found" in missing["message"]
    assert silent["success"] is False and "ref_text" in silent["message"]
    assert server.calls == []


@pytest.mark.asyncio
async def test_register_voice_needs_an_openai_compatible_speech_block(monkeypatch, voices_dir):
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)

    wrong_type = await register_voice("groq_tts", {"type": "groq", "tts_voices_dir": str(voices_dir)}, file="anna.wav", ref_text="t")
    no_url = await register_voice("fish_tts", {"type": "openai", "tts_voices_dir": str(voices_dir)}, file="anna.wav", ref_text="t")

    assert wrong_type["success"] is False and "type: openai" in wrong_type["message"]
    assert no_url["success"] is False and "tts_base_url" in no_url["message"]
    assert server.calls == []


def test_resolve_sample_path_checks_directory_format_and_size(tmp_path):
    with pytest.raises(voices.VoiceRegistrationError, match="not available"):
        resolve_sample_path(str(tmp_path / "absent"), "anna.wav")

    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(voices.VoiceRegistrationError, match="Unsupported audio format"):
        resolve_sample_path(str(tmp_path), "notes.txt")

    big = tmp_path / "big.wav"
    with open(big, "wb") as fh:
        fh.truncate(voices.MAX_SAMPLE_BYTES + 1)
    with pytest.raises(voices.VoiceRegistrationError, match="at most 10 MB"):
        resolve_sample_path(str(tmp_path), "big.wav")

    (tmp_path / "ok.flac").write_bytes(b"fLaC")
    assert resolve_sample_path(str(tmp_path), "ok.flac").name == "ok.flac"
