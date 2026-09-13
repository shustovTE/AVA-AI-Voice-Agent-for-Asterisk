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

    def post(self, url, json=None, headers=None, params=None, proxy=None, proxy_headers=None):
        self.requests.append(
            {
                "url": url,
                "params": params,
                "json": json,
                "proxy": proxy,
                "proxy_headers": proxy_headers,
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
