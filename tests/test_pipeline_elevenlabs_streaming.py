"""ElevenLabs TTS streams audio while the response body is still arriving.

The adapter used to buffer the whole sentence before the first frame reached
the caller, which put the full synthesis latency in front of every reply.
"""
import pytest

from src.audio.resampler import convert_pcm16le_to_target_format, mulaw_to_pcm16le
from src.config import AppConfig, ElevenLabsProviderConfig
from src.pipelines.elevenlabs import ElevenLabsTTSAdapter


def _app_config() -> AppConfig:
    return AppConfig(
        default_provider="elevenlabs_tts",
        providers={"elevenlabs_tts": {"api_key": "test-key"}},
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "prompt", "model": "gpt-4o"},
        audio_transport="audiosocket",
        downstream_mode="stream",
    )


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAudioResponse:
    def __init__(self, chunks, status: int = 200):
        self.status = status
        self._chunks = list(chunks)
        self.content = _FakeContent(self._chunks)
        self.read_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def read(self):
        self.read_called = True
        return b"".join(self._chunks)

    async def text(self):
        return ""

    def raise_for_status(self):
        raise AssertionError("unexpected error status")


class _FakeHttpSession:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.closed = False

    def post(self, url, json=None, headers=None, params=None, proxy=None, proxy_headers=None, timeout=None, **kwargs):
        self.requests.append(
            {
                "url": url,
                "params": params,
                "json": json,
                "proxy": proxy,
                "proxy_headers": proxy_headers,
                "timeout": timeout,
            }
        )
        return self.response

    async def close(self):
        self.closed = True


def _adapter(session, provider_config=None, options=None):
    return ElevenLabsTTSAdapter(
        "elevenlabs_tts",
        _app_config(),
        provider_config or ElevenLabsProviderConfig(api_key="test-key"),
        options or {},
        session_factory=lambda: session,
    )


async def _collect(adapter, options):
    return [chunk async for chunk in adapter.synthesize("call-1", "привет", options)]


@pytest.mark.asyncio
async def test_mulaw_stream_emits_frames_before_the_body_ends():
    # Irregular chunk sizes: the API does not align its chunks to 20 ms frames.
    chunks = [bytes([0x7F]) * 100, bytes([0x55]) * 260, bytes([0x2A]) * 40]
    response = _FakeAudioResponse(chunks)
    session = _FakeHttpSession(response)
    adapter = _adapter(session)

    frames = await _collect(
        adapter, {"format": {"encoding": "mulaw", "sample_rate": 8000}}
    )

    assert session.requests[0]["url"].endswith("/stream")
    assert response.read_called is False
    # 20 ms of μ-law at 8 kHz is 160 bytes; only the tail may be short.
    assert [len(f) for f in frames[:-1]] == [160] * (len(frames) - 1)
    expected = convert_pcm16le_to_target_format(
        mulaw_to_pcm16le(b"".join(chunks)), "mulaw"
    )
    assert b"".join(frames) == expected


@pytest.mark.asyncio
async def test_pcm_sample_split_across_chunks_is_not_corrupted():
    # Odd-length chunks force a PCM16 sample to straddle a chunk boundary.
    chunks = [b"\x01\x02\x03", b"\x04\x05", b"\x06\x07\x08"]
    response = _FakeAudioResponse(chunks)
    session = _FakeHttpSession(response)
    adapter = _adapter(session)

    frames = await _collect(
        adapter,
        {
            "output_format": "pcm_16000",
            "format": {"encoding": "linear16", "sample_rate": 16000},
        },
    )

    assert session.requests[0]["url"].endswith("/stream")
    assert b"".join(frames) == b"".join(chunks)


@pytest.mark.asyncio
async def test_resampling_output_keeps_the_buffered_path():
    response = _FakeAudioResponse([b"\x00\x01" * 240])
    session = _FakeHttpSession(response)
    adapter = _adapter(session)

    await _collect(
        adapter,
        {
            "output_format": "pcm_24000",
            "format": {"encoding": "mulaw", "sample_rate": 8000},
        },
    )

    assert not session.requests[0]["url"].endswith("/stream")
    assert response.read_called is True


@pytest.mark.asyncio
async def test_stream_can_be_disabled_per_pipeline():
    response = _FakeAudioResponse([bytes([0x7F]) * 160])
    session = _FakeHttpSession(response)
    adapter = _adapter(session, options={"stream": False})

    await _collect(
        adapter, {"format": {"encoding": "mulaw", "sample_rate": 8000}}
    )

    assert not session.requests[0]["url"].endswith("/stream")
    assert response.read_called is True


# --- output formats ---------------------------------------------------------------
#
# The adapter decodes every raw format the API documents (16-bit PCM at any
# rate, μ-law and A-law at 8 kHz); mp3 and opus are refused before a request.

import audioop

from src.pipelines.elevenlabs import _OUTPUT_FORMAT_SAMPLE_RATES


@pytest.mark.asyncio
async def test_pcm_8000_streams_on_a_narrowband_call():
    chunks = [b"\x10\x00\xf0\xff" * 50, b"\x00\x20" * 81, b"\x00\x40" * 40]
    response = _FakeAudioResponse(chunks)
    session = _FakeHttpSession(response)
    adapter = _adapter(session)

    frames = await _collect(
        adapter,
        {"output_format": "pcm_8000", "format": {"encoding": "mulaw", "sample_rate": 8000}},
    )

    assert session.requests[0]["url"].endswith("/stream")
    assert session.requests[0]["params"]["output_format"] == "pcm_8000"
    assert response.read_called is False
    assert [len(f) for f in frames[:-1]] == [160] * (len(frames) - 1)
    assert b"".join(frames) == convert_pcm16le_to_target_format(b"".join(chunks), "mulaw")


@pytest.mark.asyncio
async def test_alaw_8000_is_decoded_chunk_by_chunk():
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in range(-8000, 8000, 50))
    alaw = audioop.lin2alaw(pcm, 2)
    chunks = [alaw[:37], alaw[37:200], alaw[200:]]
    response = _FakeAudioResponse(chunks)
    session = _FakeHttpSession(response)
    adapter = _adapter(session)

    frames = await _collect(
        adapter,
        {"output_format": "alaw_8000", "format": {"encoding": "linear16", "sample_rate": 8000}},
    )

    assert session.requests[0]["params"]["output_format"] == "alaw_8000"
    assert b"".join(frames) == audioop.alaw2lin(alaw, 2)


@pytest.mark.asyncio
async def test_alaw_8000_is_decoded_on_the_buffered_path():
    pcm = b"\x00\x10" * 400
    response = _FakeAudioResponse([audioop.lin2alaw(pcm, 2)])
    session = _FakeHttpSession(response)
    adapter = _adapter(session, options={"stream": False})

    frames = await _collect(
        adapter,
        {"output_format": "alaw_8000", "format": {"encoding": "linear16", "sample_rate": 8000}},
    )

    assert response.read_called is True
    assert b"".join(frames) == audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2)


@pytest.mark.asyncio
async def test_every_documented_raw_format_is_requested_as_configured():
    assert set(_OUTPUT_FORMAT_SAMPLE_RATES) == {
        "pcm_8000", "pcm_16000", "pcm_22050", "pcm_24000", "pcm_32000",
        "pcm_44100", "pcm_48000", "ulaw_8000", "alaw_8000",
    }
    for output_format in _OUTPUT_FORMAT_SAMPLE_RATES:
        response = _FakeAudioResponse([b"\x00\x01" * 480])
        session = _FakeHttpSession(response)
        adapter = _adapter(session)

        frames = await _collect(
            adapter,
            {"output_format": output_format, "format": {"encoding": "mulaw", "sample_rate": 8000}},
        )

        assert session.requests[0]["params"]["output_format"] == output_format, output_format
        assert b"".join(frames), output_format


@pytest.mark.asyncio
async def test_compressed_formats_are_refused_before_any_request():
    for output_format in ("mp3_44100_128", "opus_48000_64"):
        session = _FakeHttpSession(_FakeAudioResponse([b"\x00"]))
        adapter = _adapter(session)

        with pytest.raises(RuntimeError) as excinfo:
            await _collect(
                adapter,
                {"output_format": output_format, "format": {"encoding": "mulaw", "sample_rate": 8000}},
            )

        message = str(excinfo.value)
        assert message.startswith(f"Unsupported ElevenLabs TTS output format: {output_format}")
        assert "decoder" in message
        assert "pcm_8000" in message and "alaw_8000" in message
        assert session.requests == []


@pytest.mark.asyncio
async def test_an_unknown_format_is_refused_with_the_accepted_list():
    session = _FakeHttpSession(_FakeAudioResponse([b"\x00"]))
    adapter = _adapter(session)

    with pytest.raises(RuntimeError, match=r"pcm_1600 \(not an output format this adapter can decode\); use one of: pcm_8000"):
        await _collect(
            adapter,
            {"output_format": "pcm_1600", "format": {"encoding": "mulaw", "sample_rate": 8000}},
        )
    assert session.requests == []


@pytest.mark.asyncio
async def test_any_8khz_format_is_raised_to_pcm_16000_on_a_wideband_call():
    for output_format in ("ulaw_8000", "alaw_8000", "pcm_8000"):
        response = _FakeAudioResponse([b"\x00\x01" * 640])
        session = _FakeHttpSession(response)
        adapter = _adapter(session)

        await _collect(
            adapter,
            {"output_format": output_format, "format": {"encoding": "linear16", "sample_rate": 16000}},
        )

        assert session.requests[0]["params"]["output_format"] == "pcm_16000", output_format
        assert session.requests[0]["url"].endswith("/stream")


# ── A stream that goes quiet ──────────────────────────────────────────────────
import asyncio

from src.pipelines.base import TTSUnavailable


class _StallingContent(_FakeContent):
    """The body stops after ``chunks``: aiohttp's read timeout then fires."""

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk
        raise asyncio.TimeoutError()


class _StallingResponse(_FakeAudioResponse):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.content = _StallingContent(chunks)


class _SequenceSession(_FakeHttpSession):
    """Serves one response per request, in order, and counts pool resets."""

    def __init__(self, responses):
        super().__init__(responses[0])
        self.responses = list(responses)
        self.close_calls = 0

    def post(self, *args, **kwargs):
        self.response = self.responses.pop(0)
        return super().post(*args, **kwargs)

    async def close(self):
        self.close_calls += 1
        await super().close()


MULAW = {"format": {"encoding": "mulaw", "sample_rate": 8000}}


@pytest.mark.asyncio
async def test_a_stream_that_never_starts_is_retried_once_on_a_fresh_connection():
    good = bytes([0x7F]) * 320
    session = _SequenceSession([_StallingResponse([]), _FakeAudioResponse([good])])
    adapter = _adapter(session)

    frames = await _collect(adapter, MULAW)

    assert len(session.requests) == 2
    assert session.close_calls == 1  # the pool was dropped before the retry
    assert b"".join(frames) == convert_pcm16le_to_target_format(mulaw_to_pcm16le(good), "mulaw")
    assert session.requests[0]["timeout"].sock_read == 8.0  # the default read timeout
    assert session.requests[0]["timeout"].total is None


@pytest.mark.asyncio
async def test_a_stream_that_stalls_after_its_first_bytes_fails_the_reply_without_a_retry():
    session = _SequenceSession([_StallingResponse([bytes([0x7F]) * 160])])
    adapter = _adapter(session)

    with pytest.raises(TTSUnavailable):
        await _collect(adapter, MULAW)

    assert len(session.requests) == 1
    assert session.close_calls == 0


@pytest.mark.asyncio
async def test_two_dead_streams_in_a_row_fail_the_reply():
    session = _SequenceSession([_StallingResponse([]), _StallingResponse([])])
    adapter = _adapter(session)

    with pytest.raises(TTSUnavailable):
        await _collect(adapter, MULAW)

    assert len(session.requests) == 2


@pytest.mark.asyncio
async def test_the_read_timeout_can_be_switched_off():
    session = _FakeHttpSession(_FakeAudioResponse([bytes([0x7F]) * 160]))
    adapter = _adapter(session, provider_config=ElevenLabsProviderConfig(api_key="test-key", read_timeout_sec=0))

    await _collect(adapter, MULAW)

    assert session.requests[0]["timeout"] is None
