"""Registering and listing reference voices on a self-hosted speech endpoint from the engine host."""
import json

import aiohttp
import pytest

from src.probes import voices
from src.probes.voices import list_voices, register_voice, register_voice_bytes, resolve_sample_path, take_staged_sample


BLOCK = {
    "type": "openai",
    "api_key": "local",
    "tts_base_url": "http://tts.lan:8091/v1/audio/speech",
    "tts_model": "fishaudio/s2-pro",
}
SAMPLE = b"RIFF" + b"\x00" * 2000


class _Server:
    """Answers the registry endpoint as scripted and records what it was sent."""

    def __init__(self, status=200, body=None, exc=None):
        self.calls = []
        self.status = status
        self.body = body if body is not None else {"success": True, "voice": {"name": "anna", "created_at": 1}}
        self.exc = exc

    async def __call__(self, url, form, headers, timeout):
        fields = {}
        for options, _headers, value in form._fields:
            fields[options["name"]] = value
            if options["name"] == "audio_sample":
                fields["_filename"] = options.get("filename")
        self.calls.append({"url": url, "fields": fields, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.status, json.dumps(self.body)


class _Listing:
    def __init__(self, status=200, body=None, exc=None):
        self.calls = []
        self.status = status
        self.body = body
        self.exc = exc

    async def __call__(self, url, headers, timeout):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.status, self.body if isinstance(self.body, str) else json.dumps(self.body)


class _Unreachable(aiohttp.ClientConnectorError):
    def __init__(self):
        Exception.__init__(self, "refused")

    def __str__(self):
        return "Connection refused"


@pytest.mark.asyncio
async def test_register_voice_bytes_uploads_the_sample_with_its_transcript(monkeypatch):
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)

    result = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="anna.wav", ref_text="Здравствуйте, это Анна.")

    assert result["success"] is True
    assert result["voice"] == "anna"
    assert "registered at tts.lan" in result["message"]
    call = server.calls[0]
    assert call["url"] == "http://tts.lan:8091/v1/audio/voices"
    assert call["headers"] == {"Authorization": "Bearer local"}
    assert call["fields"]["_filename"] == "anna.wav"
    assert call["fields"]["audio_sample"] == SAMPLE
    assert call["fields"]["name"] == "anna"
    assert call["fields"]["ref_text"] == "Здравствуйте, это Анна."
    assert call["fields"]["consent"] == "ava-admin"


@pytest.mark.asyncio
async def test_register_voice_bytes_takes_an_explicit_name_and_consent(monkeypatch):
    server = _Server(body={"success": True, "voice": {"name": "Anna-RU"}})
    monkeypatch.setattr(voices, "_post_form", server)

    result = await register_voice_bytes(
        "fish_tts", BLOCK, sample=SAMPLE, filename="anna.wav", name="Anna-RU", ref_text="text", consent="consent-2026"
    )

    assert result["voice"] == "Anna-RU"
    assert server.calls[0]["fields"]["name"] == "Anna-RU"
    assert server.calls[0]["fields"]["consent"] == "consent-2026"


@pytest.mark.asyncio
async def test_register_voice_bytes_relays_the_servers_refusal(monkeypatch):
    server = _Server(status=400, body={"error": {"message": "Reference audio too short (0.5s).", "type": "BadRequestError"}})
    monkeypatch.setattr(voices, "_post_form", server)

    result = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="anna.wav", ref_text="text")

    assert result["success"] is False
    assert "HTTP 400: Reference audio too short (0.5s)." in result["message"]


@pytest.mark.asyncio
async def test_register_voice_bytes_names_an_unreachable_server(monkeypatch):
    monkeypatch.setattr(voices, "_post_form", _Server(exc=_Unreachable()))

    result = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="anna.wav", ref_text="text")

    assert result["success"] is False
    assert result["message"].startswith("Cannot connect to the speech server at http://tts.lan:8091/v1")


@pytest.mark.asyncio
async def test_register_voice_bytes_refuses_what_the_server_would_refuse(monkeypatch):
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)

    empty = await register_voice_bytes("fish_tts", BLOCK, sample=b"", filename="anna.wav", ref_text="t")
    huge = await register_voice_bytes("fish_tts", BLOCK, sample=b"x" * (voices.MAX_SAMPLE_BYTES + 1), filename="anna.wav", ref_text="t")
    fmt = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="notes.txt", ref_text="t")
    path = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="../anna.wav", ref_text="t")
    silent = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="anna.wav", ref_text="   ")
    bad_name = await register_voice_bytes("fish_tts", BLOCK, sample=SAMPLE, filename="anna.wav", name="a/b", ref_text="t")

    assert empty["success"] is False and "empty" in empty["message"]
    assert huge["success"] is False and "at most 10 MB" in huge["message"]
    assert fmt["success"] is False and "Unsupported audio format" in fmt["message"]
    assert path["success"] is False and "path separators" in path["message"]
    assert silent["success"] is False and "ref_text" in silent["message"]
    assert bad_name["success"] is False and "Voice name" in bad_name["message"]
    assert server.calls == []


@pytest.mark.asyncio
async def test_register_voice_bytes_needs_an_openai_compatible_speech_block(monkeypatch):
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)

    wrong_type = await register_voice_bytes("groq_tts", {"type": "groq"}, sample=SAMPLE, filename="a.wav", ref_text="t")
    no_url = await register_voice_bytes("fish_tts", {"type": "openai"}, sample=SAMPLE, filename="a.wav", ref_text="t")

    assert wrong_type["success"] is False and "type: openai" in wrong_type["message"]
    assert no_url["success"] is False and "tts_base_url" in no_url["message"]
    assert server.calls == []


@pytest.mark.asyncio
async def test_register_voice_reads_the_file_from_the_voices_directory(monkeypatch, tmp_path):
    (tmp_path / "anna.wav").write_bytes(SAMPLE)
    server = _Server()
    monkeypatch.setattr(voices, "_post_form", server)
    block = {**BLOCK, "tts_voices_dir": str(tmp_path)}

    result = await register_voice("fish_tts", block, file="anna.wav", ref_text="text")
    traversal = await register_voice("fish_tts", block, file="../etc/passwd", ref_text="text")
    missing = await register_voice("fish_tts", block, file="nobody.wav", ref_text="text")

    assert result["success"] is True
    assert server.calls[0]["fields"]["audio_sample"] == SAMPLE
    assert traversal["success"] is False and "path separators" in traversal["message"]
    assert missing["success"] is False and "not found" in missing["message"]
    assert len(server.calls) == 1


def test_take_staged_sample_reads_then_deletes(monkeypatch, tmp_path):
    monkeypatch.setattr(voices, "STAGING_DIR", tmp_path)
    (tmp_path / "abc.wav").write_bytes(SAMPLE)

    data, name = take_staged_sample("abc.wav")

    assert data == SAMPLE and name == "abc.wav"
    assert not (tmp_path / "abc.wav").exists()
    with pytest.raises(voices.VoiceRegistrationError, match="not found"):
        take_staged_sample("abc.wav")
    with pytest.raises(voices.VoiceRegistrationError, match="path separators"):
        take_staged_sample("../abc.wav")


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


@pytest.mark.asyncio
async def test_list_voices_reports_registered_and_builtin_voices(monkeypatch):
    listing = _Listing(
        body={
            "voices": ["default", "anna"],
            "uploaded_voices": [
                {"name": "anna", "consent": "c", "created_at": 1727000000, "file_size": 320000, "mime_type": "audio/wav", "ref_text": "Привет"},
                {"name": "", "created_at": 1},
            ],
        }
    )
    monkeypatch.setattr(voices, "_http_get", listing)

    result = await list_voices("fish_tts", BLOCK)

    assert result["success"] is True
    assert listing.calls[0]["url"] == "http://tts.lan:8091/v1/audio/voices"
    assert listing.calls[0]["headers"] == {"Authorization": "Bearer local"}
    assert result["voices"] == [
        {"name": "anna", "created_at": 1727000000, "file_size": 320000, "ref_text": "Привет", "consent": "c", "speaker_description": None}
    ]
    assert result["builtin"] == ["default", "anna"]
    assert result["message"] == "1 registered voice(s) on tts.lan; built-in: default, anna"


@pytest.mark.asyncio
async def test_list_voices_reports_server_trouble(monkeypatch):
    monkeypatch.setattr(voices, "_http_get", _Listing(status=404, body={"error": {"message": "The model does not support Speech API"}}))
    refused = await list_voices("fish_tts", BLOCK)

    monkeypatch.setattr(voices, "_http_get", _Listing(body="<html>not json</html>"))
    garbage = await list_voices("fish_tts", BLOCK)

    monkeypatch.setattr(voices, "_http_get", _Listing(exc=_Unreachable()))
    down = await list_voices("fish_tts", BLOCK)

    assert refused["success"] is False and "HTTP 404: The model does not support Speech API" in refused["message"]
    assert garbage["success"] is False and "something other than a voice list" in garbage["message"]
    assert down["success"] is False and down["message"].startswith("Cannot connect to the speech server")
