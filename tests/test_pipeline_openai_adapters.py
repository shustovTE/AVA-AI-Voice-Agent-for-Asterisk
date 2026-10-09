import asyncio
import audioop
import base64
import json
import wave
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from src.audio.resampler import StreamingResampler, convert_pcm16le_to_target_format
from src.config import AppConfig, OpenAIProviderConfig
from src.pipelines import openai as openai_module
from src.pipelines.openai import OpenAISTTAdapter, OpenAILLMAdapter, OpenAITTSAdapter
from src.pipelines.orchestrator import PipelineOrchestrator
from src.tools.base import ToolPhase


def _build_app_config() -> AppConfig:
    providers = {
        "openai": {
            "api_key": "test-key",
            "organization": "test-org",
            "project": "test-project",
            "realtime_base_url": "wss://api.openai.com/v1/realtime",
            "chat_base_url": "https://api.openai.com/v1",
            "tts_base_url": "https://api.openai.com/v1/audio/speech",
            "realtime_model": "gpt-realtime",
            "chat_model": "gpt-4o-mini",
            "tts_model": "gpt-4o-mini-tts",
            "voice": "alloy",
            "default_modalities": ["text"],
            "input_encoding": "linear16",
            "input_sample_rate_hz": 8000,
            "target_encoding": "mulaw",
            "target_sample_rate_hz": 8000,
            "chunk_size_ms": 20,
            "response_timeout_sec": 2.0,
        }
    }
    pipelines = {
        "openai_stack": {
            "stt": "openai_stt",
            "llm": "openai_llm",
            "tts": "openai_tts",
            "options": {
                "stt": {},
                "llm": {"use_realtime": False, "temperature": 0.5},
                "tts": {"format": {"encoding": "mulaw", "sample_rate": 8000}},
            },
        }
    }
    return AppConfig(
        default_provider="openai",
        providers=providers,
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "prompt", "model": "gpt-4o"},
        audio_transport="audiosocket",
        downstream_mode="stream",
        pipelines=pipelines,
        active_pipeline="openai_stack",
    )


class _MockWebSocket:
    def __init__(self):
        self.sent = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return await self._queue.get()

    async def close(self):
        self.closed = True

    def push(self, message):
        self._queue.put_nowait(message)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def read(self):
        return self._body

    async def text(self):
        return self._body.decode("utf-8", errors="ignore")


class _FakeSession:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self._status = status
        self.requests = []
        self.closed = False

    def post(self, url, json=None, data=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "data": data, "headers": headers, "timeout": timeout})
        return _FakeResponse(self._body, status=self._status)

    async def close(self):
        self.closed = True


class _TimeoutStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise asyncio.TimeoutError("simulated first-token timeout")


class _FakeStreamingResponse(_FakeResponse):
    def __init__(self, content, status: int = 200):
        super().__init__(b"", status=status)
        self.content = content


class _FakeStreamingSession(_FakeSession):
    def __init__(self, content, status: int = 200):
        super().__init__(b"", status=status)
        self._content = content

    def post(self, url, json=None, data=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "data": data, "headers": headers, "timeout": timeout})
        return _FakeStreamingResponse(self._content, status=self._status)


@pytest.mark.asyncio
async def test_openai_stt_adapter_transcribes(monkeypatch):
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    body = json.dumps({"text": "hello"}).encode("utf-8")
    fake_session = _FakeSession(body)
    adapter = OpenAISTTAdapter(
        "openai_stt",
        app_config,
        provider_config,
        {},
        session_factory=lambda: fake_session,
    )

    await adapter.start()
    await adapter.open_call("call-1", {})

    audio_buffer = b"\x00\x10" * 160  # 10 ms @ 16 kHz
    transcript = await adapter.transcribe("call-1", audio_buffer, 16000, {})
    assert transcript == "hello"
    request = fake_session.requests[0]
    assert request["url"].endswith("/audio/transcriptions")
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["data"] is not None


@pytest.mark.asyncio
async def test_openai_llm_adapter_chat_completion(monkeypatch):
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    body = json.dumps({"choices": [{"message": {"content": "hi there"}}]}).encode("utf-8")
    fake_session = _FakeSession(body)

    adapter = OpenAILLMAdapter(
        "openai_llm",
        app_config,
        provider_config,
        {"use_realtime": False},
        session_factory=lambda: fake_session,
    )

    await adapter.start()
    summary_prompt = "Summarize the call. Do not use the agent persona."
    response = await adapter.generate(
        "call-1",
        "hello",
        {"system_prompt": summary_prompt},
        {"system_prompt": summary_prompt, "instructions": summary_prompt},
    )
    assert response.text == "hi there"

    request = fake_session.requests[0]
    assert request["json"]["model"] == "gpt-4o-mini"
    assert request["json"]["messages"][0] == {"role": "system", "content": summary_prompt}
    assert request["json"]["messages"][-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_openai_llm_uses_captured_registry_for_tool_schemas(monkeypatch):
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
    fake_session = _FakeSession(body)
    adapter = OpenAILLMAdapter(
        "openai_llm",
        app_config,
        provider_config,
        {"use_realtime": False},
        session_factory=lambda: fake_session,
    )
    definition = MagicMock()
    definition.phase = ToolPhase.IN_CALL
    definition.to_openai_schema.return_value = {
        "type": "function",
        "function": {
            "name": "captured_tool",
            "description": "Captured generation tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    captured_tool = MagicMock(definition=definition)
    captured_registry = MagicMock()
    captured_registry.get.return_value = captured_tool
    live_registry = MagicMock()
    live_registry.get.return_value = None
    adapter.bind_tool_registry(captured_registry)
    monkeypatch.setattr("src.pipelines.openai.tool_registry", live_registry)

    await adapter.start()
    await adapter.generate(
        "call-1", "hello", {}, {"tools": ["captured_tool"]}
    )

    assert fake_session.requests[0]["json"]["tools"] == [
        definition.to_openai_schema.return_value
    ]
    captured_registry.get.assert_called_once_with("captured_tool")
    live_registry.get.assert_not_called()


@pytest.mark.asyncio
async def test_openai_llm_stream_timeout_returns_empty_for_serial_fallback():
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    fake_session = _FakeStreamingSession(_TimeoutStream())
    adapter = OpenAILLMAdapter(
        "openai_llm",
        app_config,
        provider_config,
        {"use_realtime": False, "timeout_sec": 1.5},
        session_factory=lambda: fake_session,
    )

    await adapter.start()
    chunks = [chunk async for chunk in adapter.generate_stream("call-1", "hello", {"system_prompt": "You are helpful."}, {})]

    assert chunks == []
    assert fake_session.requests[0]["json"]["stream"] is True
    assert fake_session.requests[0]["timeout"] == 1.5
    assert adapter._pending_tool_calls_by_call["call-1"] == []


@pytest.mark.asyncio
async def test_openai_llm_adapter_realtime(monkeypatch):
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    adapter = OpenAILLMAdapter("openai_llm", app_config, provider_config, {"use_realtime": True})

    mock_ws = _MockWebSocket()

    async def fake_connect(*args, **kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.openai.websockets.connect", fake_connect)

    await adapter.start()
    task = asyncio.create_task(
        adapter.generate(
            "call-1",
            "hello listener",
            {"system_prompt": "You are concise."},
            {"use_realtime": True},
        )
    )

    await asyncio.sleep(0)
    mock_ws.push(json.dumps({"type": "response.output_text.delta", "delta": "response"}))
    mock_ws.push(json.dumps({"type": "response.output_text.done"}))

    result = await task
    assert result == "response"

    session_event = json.loads(mock_ws.sent[0])
    assert session_event["type"] == "session.create"
    response_event = json.loads(mock_ws.sent[1])
    assert response_event["type"] == "response.create"


@pytest.mark.asyncio
async def test_openai_tts_adapter_synthesizes_chunks():
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])

    pcm_audio = b"\x00\x10" * 160  # 20 ms @ 8 kHz
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm_audio)
    wav_bytes = buf.getvalue()
    fake_session = _FakeSession(wav_bytes)

    adapter = OpenAITTSAdapter(
        "openai_tts",
        app_config,
        provider_config,
        {"response_format": "wav", "format": {"encoding": "mulaw", "sample_rate": 8000}},
        session_factory=lambda: fake_session,
    )

    await adapter.start()
    await adapter.open_call("call-1", {})

    chunks = [chunk async for chunk in adapter.synthesize("call-1", "Hello caller", {})]
    synthesized = b"".join(chunks)
    expected = convert_pcm16le_to_target_format(pcm_audio, "mulaw")

    assert synthesized == expected
    request = fake_session.requests[0]
    assert request["json"]["model"] == "gpt-4o-mini-tts"
    assert request["json"]["voice"] == "alloy"
    assert request["json"]["response_format"] == "wav"


@pytest.mark.asyncio
async def test_pipeline_orchestrator_registers_openai_adapters():
    app_config = _build_app_config()
    orchestrator = PipelineOrchestrator(app_config)
    await orchestrator.start()

    resolution = orchestrator.get_pipeline("call-1")
    assert isinstance(resolution.stt_adapter, OpenAISTTAdapter)
    assert isinstance(resolution.llm_adapter, OpenAILLMAdapter)
    assert isinstance(resolution.tts_adapter, OpenAITTSAdapter)
    assert resolution.tts_options["format"]["encoding"] == "mulaw"


class _LinesStream:
    """An SSE body: yields the given lines as the aiohttp content iterator would."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


def _sse(chunk: dict) -> bytes:
    return ("data: " + json.dumps(chunk) + "\n").encode("utf-8")


def _capture_logs(monkeypatch, level: str):
    calls = []
    monkeypatch.setattr(
        openai_module.logger, level, lambda event, **kw: calls.append((event, kw))
    )
    return calls


@pytest.mark.asyncio
async def test_openai_llm_stream_warns_when_the_reply_is_cut_by_max_tokens(monkeypatch):
    """A reply that stops on max_tokens is spoken mid-sentence; the log must say so."""
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    lines = [
        _sse({"choices": [{"delta": {"content": "Если "}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "хотите"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        b"data: [DONE]\n",
    ]
    fake_session = _FakeStreamingSession(_LinesStream(lines))
    adapter = OpenAILLMAdapter(
        "openai_llm",
        app_config,
        provider_config,
        {"use_realtime": False, "max_tokens": 200},
        session_factory=lambda: fake_session,
    )
    warnings = _capture_logs(monkeypatch, "warning")

    await adapter.start()
    chunks = [chunk async for chunk in adapter.generate_stream("call-1", "hello", {}, {})]

    assert "".join(chunks) == "Если хотите"
    cut = [kw for event, kw in warnings if event == "LLM reply cut by max_tokens"]
    assert len(cut) == 1
    assert cut[0]["max_tokens"] == 200
    assert cut[0]["chars"] == len("Если хотите")


@pytest.mark.asyncio
async def test_openai_llm_stream_stays_quiet_on_a_natural_stop(monkeypatch):
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    lines = [
        _sse({"choices": [{"delta": {"content": "Готово."}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n",
    ]
    fake_session = _FakeStreamingSession(_LinesStream(lines))
    adapter = OpenAILLMAdapter(
        "openai_llm",
        app_config,
        provider_config,
        {"use_realtime": False},
        session_factory=lambda: fake_session,
    )
    warnings = _capture_logs(monkeypatch, "warning")

    await adapter.start()
    chunks = [chunk async for chunk in adapter.generate_stream("call-1", "hello", {}, {})]

    assert "".join(chunks) == "Готово."
    assert not [kw for event, kw in warnings if event == "LLM reply cut by max_tokens"]


@pytest.mark.asyncio
async def test_openai_llm_generate_warns_when_the_reply_is_cut_by_max_tokens(monkeypatch):
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    body = json.dumps(
        {
            "choices": [
                {"message": {"content": "входная дверь, ме"}, "finish_reason": "length"}
            ]
        }
    ).encode("utf-8")
    fake_session = _FakeSession(body)
    adapter = OpenAILLMAdapter(
        "openai_llm",
        app_config,
        provider_config,
        {"use_realtime": False, "max_tokens": 150},
        session_factory=lambda: fake_session,
    )
    warnings = _capture_logs(monkeypatch, "warning")
    infos = _capture_logs(monkeypatch, "info")

    await adapter.start()
    response = await adapter.generate("call-1", "hello", {}, {})

    assert response.text == "входная дверь, ме"
    cut = [kw for event, kw in warnings if event == "LLM reply cut by max_tokens"]
    assert len(cut) == 1
    assert cut[0]["max_tokens"] == 150
    received = [kw for event, kw in infos if event == "OpenAI chat completion received"]
    assert received and received[0]["finish_reason"] == "length"


# --- OpenAI TTS adapter: streamed replies from an OpenAI-compatible endpoint -----


class _FakeStreamContent:
    """Stands in for ``aiohttp.StreamReader``: ``iter_any()`` hands out the given pieces."""

    def __init__(self, pieces):
        self._pieces = list(pieces)

    async def iter_any(self):
        for piece in self._pieces:
            await asyncio.sleep(0)
            yield piece


def _tone_pcm16(rate: int, seconds: float, freq: float = 440.0) -> bytes:
    import numpy as np

    n = int(rate * seconds)
    return (6000 * np.sin(2 * np.pi * freq * np.arange(n) / rate)).astype("<i2").tobytes()


def _split(data: bytes, sizes) -> list:
    pieces, offset = [], 0
    for size in sizes:
        pieces.append(data[offset : offset + size])
        offset += size
    if offset < len(data):
        pieces.append(data[offset:])
    return pieces


def _self_hosted_tts_config(**overrides) -> OpenAIProviderConfig:
    fields = {
        "api_key": "not-checked-by-vllm",
        "tts_base_url": "http://tts.lan:8091/v1/audio/speech",
        "tts_model": "fishaudio/s2-pro",
        "voice": "anna",
        "tts_streaming": True,
        "tts_pcm_sample_rate_hz": 16000,
        "tts_text_prefix": "<|speaker:0|>",
        "tts_extra_body": {"stream_format": "audio", "extra_params": {"top_p": 0.8, "temperature": 0.7}},
        "target_encoding": "mulaw",
        "target_sample_rate_hz": 8000,
        "chunk_size_ms": 20,
        "response_timeout_sec": 2.0,
    }
    fields.update(overrides)
    return OpenAIProviderConfig(**fields)


@pytest.mark.asyncio
async def test_openai_tts_adapter_streams_pcm_frames_from_a_self_hosted_endpoint():
    app_config = _build_app_config()
    provider_config = _self_hosted_tts_config(tts_extra_body={
        "stream_format": "audio",
        "extra_params": {"top_p": 0.8},
        "model": "must-not-override",
        "stream": False,
    })
    pcm16k = _tone_pcm16(16000, 0.25)
    # Uneven pieces, some odd-sized so a sample straddles a chunk boundary.
    pieces = _split(pcm16k, [1001, 333, 7, 2048, 999])
    fake_session = _FakeStreamingSession(_FakeStreamContent(pieces))

    adapter = OpenAITTSAdapter(
        "openai_tts",
        app_config,
        provider_config,
        {"response_format": "pcm", "format": {"encoding": "mulaw", "sample_rate": 8000}},
        session_factory=lambda: fake_session,
    )
    await adapter.start()
    await adapter.open_call("call-1", {})

    frames = [chunk async for chunk in adapter.synthesize("call-1", "Здравствуйте! Меня зовут Анна.", {})]

    request = fake_session.requests[0]["json"]
    assert request["model"] == "fishaudio/s2-pro"  # no OpenAI fallback for a self-hosted host
    assert request["voice"] == "anna"
    assert request["input"] == "<|speaker:0|>Здравствуйте! Меня зовут Анна."
    assert request["stream"] is True  # engine-owned: the extra_body value is ignored
    assert request["response_format"] == "pcm"
    assert request["stream_format"] == "audio"
    assert request["extra_params"] == {"top_p": 0.8}
    timeout = fake_session.requests[0]["timeout"]
    assert timeout.total is None and timeout.sock_read == 2.0

    # 20 ms μ-law frames at 8 kHz, only the last one may be short.
    assert frames
    assert all(len(frame) == 160 for frame in frames[:-1])
    assert 0 < len(frames[-1]) <= 160

    reference = StreamingResampler(16000, 8000, "linear")
    expected = convert_pcm16le_to_target_format(reference.process(pcm16k) + reference.flush(), "mulaw")
    produced = b"".join(frames)
    assert len(produced) == len(expected)
    got = audioop.ulaw2lin(produced, 2)
    want = audioop.ulaw2lin(expected, 2)
    diff = max(abs(a - b) for a, b in zip(
        memoryview(got).cast("h"), memoryview(want).cast("h")
    ))
    assert diff <= 64  # one μ-law step around the loudest samples


@pytest.mark.asyncio
async def test_openai_tts_adapter_streams_wav_and_reads_the_rate_from_a_split_header():
    app_config = _build_app_config()
    provider_config = _self_hosted_tts_config(tts_extra_body={})
    pcm8k = b"".join((i % 200 - 100).to_bytes(2, "little", signed=True) for i in range(800))
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm8k)
    wav_bytes = buf.getvalue()
    # The 44-byte header arrives in three pieces, the last one carrying samples.
    pieces = _split(wav_bytes, [5, 25, 300, 700])
    fake_session = _FakeStreamingSession(_FakeStreamContent(pieces))

    adapter = OpenAITTSAdapter(
        "openai_tts",
        app_config,
        provider_config,
        {"response_format": "wav", "format": {"encoding": "mulaw", "sample_rate": 8000}},
        session_factory=lambda: fake_session,
    )
    await adapter.start()
    await adapter.open_call("call-1", {})

    frames = [chunk async for chunk in adapter.synthesize("call-1", "Проверка.", {})]

    assert fake_session.requests[0]["json"]["response_format"] == "wav"
    assert fake_session.requests[0]["json"]["stream"] is True
    assert b"".join(frames) == convert_pcm16le_to_target_format(pcm8k, "mulaw")
    assert all(len(frame) == 160 for frame in frames)


@pytest.mark.asyncio
async def test_openai_tts_adapter_streaming_error_status_raises():
    app_config = _build_app_config()
    provider_config = _self_hosted_tts_config()
    fake_session = _FakeStreamingSession(_FakeStreamContent([b"unused"]), status=400)
    fake_session._body = b'{"error": "ref_text is required"}'

    adapter = OpenAITTSAdapter(
        "openai_tts",
        app_config,
        provider_config,
        {"response_format": "pcm", "format": {"encoding": "mulaw", "sample_rate": 8000}},
        session_factory=lambda: fake_session,
    )
    await adapter.start()
    await adapter.open_call("call-1", {})

    with pytest.raises(RuntimeError, match="status 400"):
        async for _chunk in adapter.synthesize("call-1", "Проверка.", {}):
            pass


@pytest.mark.asyncio
async def test_openai_tts_adapter_keeps_openai_fallbacks_for_openai_hosts():
    app_config = _build_app_config()
    provider_config = OpenAIProviderConfig(**app_config.providers["openai"])
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x10" * 160)
    fake_session = _FakeSession(buf.getvalue())

    adapter = OpenAITTSAdapter(
        "openai_tts",
        app_config,
        provider_config,
        {
            "response_format": "wav",
            "format": {"encoding": "mulaw", "sample_rate": 8000},
            "model": "canopylabs/orpheus-v1-english",  # a stale Groq model name
            "voice": "hannah",  # a stale Groq voice name
        },
        session_factory=lambda: fake_session,
    )
    await adapter.start()
    await adapter.open_call("call-1", {})
    _ = [chunk async for chunk in adapter.synthesize("call-1", "Hello", {})]

    request = fake_session.requests[0]["json"]
    assert request["model"] == "gpt-4o-mini-tts"
    assert request["voice"] == "alloy"
    assert "stream" not in request


@pytest.mark.asyncio
async def test_openai_tts_adapter_non_streaming_request_carries_prefix_and_extra_body():
    app_config = _build_app_config()
    provider_config = _self_hosted_tts_config(tts_streaming=False, tts_extra_body={"extra_params": {"top_p": 0.8}})
    pcm = b"\x00\x10" * 320  # 20 ms @ 16 kHz
    fake_session = _FakeSession(pcm)

    adapter = OpenAITTSAdapter(
        "openai_tts",
        app_config,
        provider_config,
        {"response_format": "pcm", "format": {"encoding": "mulaw", "sample_rate": 8000}},
        session_factory=lambda: fake_session,
    )
    await adapter.start()
    await adapter.open_call("call-1", {})
    frames = [chunk async for chunk in adapter.synthesize("call-1", "<|speaker:0|>Уже с тегом.", {})]

    request = fake_session.requests[0]["json"]
    assert request["input"] == "<|speaker:0|>Уже с тегом."  # the prefix is not doubled
    assert request["extra_params"] == {"top_p": 0.8}
    assert "stream" not in request
    assert len(b"".join(frames)) == 160  # 16 kHz pcm brought to 8 kHz μ-law
