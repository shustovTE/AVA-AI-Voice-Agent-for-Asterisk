import asyncio
import base64
import json

import pytest

from src.config import AppConfig, LocalProviderConfig
from src.pipelines.local import LocalLLMAdapter, LocalSTTAdapter, LocalTTSAdapter
from src.pipelines.orchestrator import PipelineOrchestrator


def _build_app_config() -> AppConfig:
    providers = {
        "local": {
            "enabled": True,
            "ws_url": "ws://127.0.0.1:8765",
            "connect_timeout_sec": 0.5,
            "response_timeout_sec": 0.5,
            "chunk_ms": 200,
        }
    }
    pipelines = {
        "local_only": {
            "stt": "local_stt",
            "llm": "local_llm",
            "tts": "local_tts",
            "options": {
                "stt": {"mode": "stt"},
                "llm": {"mode": "llm"},
                "tts": {"mode": "tts"},
            },
        }
    }
    return AppConfig(
        default_provider="local",
        providers=providers,
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "prompt", "model": "local-llm"},
        audio_transport="audiosocket",
        downstream_mode="file",
        pipelines=pipelines,
        active_pipeline="local_only",
    )


class _MockState:
    """Mock websockets State enum."""
    name = "OPEN"


class _MockWebSocket:
    def __init__(self):
        self.sent = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.state = _MockState()

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        self.closed = True

    def push(self, message):
        self._queue.put_nowait(message)


async def _collect_async(iterator):
    return [item async for item in iterator]


def test_local_stt_adapter_names_the_backends_that_cannot_run_buffered():
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})
    assert adapter.requires_streaming() is False  # vosk answers per chunk

    for backend in ("onnx_asr", "tone"):
        adapter = LocalSTTAdapter(
            "local_stt", app_config, provider_config.model_copy(update={"stt_backend": backend}), {"mode": "stt"}
        )
        assert adapter.requires_streaming() is True
        assert adapter.requires_streaming({"streaming": False}) is True
    # A per-pipeline override of the backend is honoured too.
    assert adapter.requires_streaming({"stt_backend": "vosk"}) is False
    assert LocalSTTAdapter("local_stt", app_config, provider_config, {}).requires_streaming({"stt_backend": "ONNX_ASR"}) is True


@pytest.mark.asyncio
async def test_local_stt_adapter_transcribes(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})

    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start()
    await adapter.open_call("call-1", {"mode": "stt"})

    set_mode_message = json.loads(mock_ws.sent[0])
    assert set_mode_message == {"type": "set_mode", "mode": "stt", "call_id": "call-1"}

    audio_buffer = b"\x01\x02" * 80  # 160 bytes == 20 ms of 8 kHz PCM16
    task = asyncio.create_task(adapter.transcribe("call-1", audio_buffer, 8000, {}))
    await asyncio.sleep(0)

    partial_payload = {
        "type": "stt_result",
        "text": "hello",
        "is_partial": True,
        "is_final": False,
    }
    final_payload = {
        "type": "stt_result",
        "text": "hello world",
        "is_partial": False,
        "is_final": True,
    }

    mock_ws.push(json.dumps(partial_payload))
    await asyncio.sleep(0)
    mock_ws.push(json.dumps(final_payload))

    transcript = await task
    assert transcript == "hello world"

    audio_message = json.loads(mock_ws.sent[1])
    assert audio_message["type"] == "audio"
    assert audio_message["mode"] == "stt"
    decoded = base64.b64decode(audio_message["data"])
    assert decoded == audio_buffer


@pytest.mark.asyncio
async def test_local_stt_adapter_sends_pipeline_segmenter_overrides(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter(
        "local_stt",
        app_config,
        provider_config,
        {
            "mode": "stt",
            "segment_energy_threshold": 800,
            "segment_silence_ms": 1200,
        },
    )
    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.open_call("call-segment-policy", {})

    assert json.loads(mock_ws.sent[0]) == {
        "type": "set_mode",
        "mode": "stt",
        "call_id": "call-segment-policy",
        "segment_energy_threshold": 800,
        "segment_silence_ms": 1200,
    }
    await adapter.close_call("call-segment-policy")


@pytest.mark.asyncio
async def test_local_stt_stream_accepts_linear16_alias(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})

    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start_stream(
        "call-linear16",
        {"mode": "stt"},
        sample_rate_hz=16000,
        fmt="linear16",
    )
    try:
        audio_buffer = b"\x01\x02" * 160
        await adapter.send_audio("call-linear16", audio_buffer, fmt="linear16")

        audio_message = json.loads(mock_ws.sent[-1])
        assert audio_message["type"] == "audio"
        assert audio_message["mode"] == "stt"
        assert audio_message["rate"] == 16000
        assert audio_message["format"] == "pcm16le"
        assert base64.b64decode(audio_message["data"]) == audio_buffer
    finally:
        await adapter.close_call("call-linear16")


@pytest.mark.asyncio
async def test_local_stt_stream_recovers_receiver_without_ending_results(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})
    mock_ws = _MockWebSocket()
    mock_ws.push(json.dumps({"type": "mode_ready", "mode": "stt", "call_id": "call-recover"}))

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start_stream(
        "call-recover",
        {"mode": "stt"},
        sample_rate_hz=16000,
        fmt="linear16",
    )
    session = adapter._sessions["call-recover"]
    first_receiver = session.receiver_task
    mock_ws.push(RuntimeError("transient receive failure"))
    await asyncio.wait_for(first_receiver, timeout=1)

    result_task = asyncio.create_task(anext(adapter.iter_results("call-recover")))
    await adapter.send_audio("call-recover", b"\x01\x02" * 160, fmt="linear16")
    assert session.receiver_restart_count == 1
    assert session.receiver_task is not first_receiver

    mock_ws.push(json.dumps({
        "type": "stt_result",
        "text": "second turn survives",
        "is_partial": False,
        "is_final": True,
    }))
    assert await asyncio.wait_for(result_task, timeout=1) == "second turn survives"
    await adapter.close_call("call-recover")


@pytest.mark.asyncio
async def test_local_stt_stream_preserves_result_queue_across_websocket_reconnect(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})
    first_ws = _MockWebSocket()
    second_ws = _MockWebSocket()
    first_ws.push(json.dumps({"type": "mode_ready", "mode": "stt", "call_id": "call-reconnect"}))
    second_ws.push(json.dumps({"type": "mode_ready", "mode": "stt", "call_id": "call-reconnect"}))
    sockets = iter((first_ws, second_ws))

    async def fake_connect(*_args, **_kwargs):
        return next(sockets)

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start_stream(
        "call-reconnect",
        {"mode": "stt"},
        sample_rate_hz=16000,
        fmt="linear16",
    )
    original = adapter._sessions["call-reconnect"]
    original_queue = original.result_queue
    result_task = asyncio.create_task(anext(adapter.iter_results("call-reconnect")))
    await asyncio.sleep(0)
    original.websocket.state.name = "CLOSED"

    await adapter.send_audio("call-reconnect", b"\x01\x02" * 160, fmt="linear16")

    reconnected = adapter._sessions["call-reconnect"]
    assert reconnected is not original
    assert reconnected.result_queue is original_queue
    second_ws.push(json.dumps({
        "type": "stt_result",
        "text": "result after reconnect",
        "is_partial": False,
        "is_final": True,
    }))
    assert await asyncio.wait_for(result_task, timeout=1) == "result after reconnect"
    await adapter.close_call("call-reconnect")


@pytest.mark.asyncio
async def test_local_llm_adapter_generate(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalLLMAdapter("local_llm", app_config, provider_config, {"mode": "llm"})

    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start()
    await adapter.open_call("call-2", {"mode": "llm"})

    request_task = asyncio.create_task(
        adapter.generate(
            "call-2",
            "user text",
            {"messages": [{"role": "user", "content": "user text"}]},
            {},
        )
    )
    await asyncio.sleep(0)

    mock_ws.push(json.dumps({"type": "llm_response", "text": "assistant reply"}))

    response = await request_task
    assert response.text == "assistant reply"

    llm_message = json.loads(mock_ws.sent[1])
    assert llm_message["type"] == "llm_request"
    assert llm_message["call_id"] == "call-2"
    assert llm_message["text"] == "user text"
    assert llm_message["context"] == [{"role": "user", "content": "user text"}]


@pytest.mark.asyncio
async def test_local_tts_adapter_synthesizes(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalTTSAdapter("local_tts", app_config, provider_config, {"mode": "tts"})

    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start()
    await adapter.open_call("call-3", {"mode": "tts"})

    audio_bytes = b"\xAA\xBB" * 40  # 80 bytes
    encoded = base64.b64encode(audio_bytes).decode("ascii")

    collected = []

    async def collect_audio():
        async for chunk in adapter.synthesize("call-3", "Hello world", {}):
            collected.append(chunk)

    task = asyncio.create_task(collect_audio())
    await asyncio.sleep(0)

    mock_ws.push(json.dumps({"type": "tts_response", "audio_data": encoded}))

    await task

    assert collected == [audio_bytes]

    tts_message = json.loads(mock_ws.sent[1])
    assert tts_message["type"] == "tts_request"
    assert tts_message["call_id"] == "call-3"
    assert tts_message["text"] == "Hello world"
    assert tts_message["output_encoding"] == "mulaw"
    assert tts_message["output_sample_rate_hz"] == 8000


@pytest.mark.asyncio
async def test_local_tts_adapter_requests_native_wideband(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalTTSAdapter("local_tts", app_config, provider_config, {"mode": "tts"})
    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)
    options = {"format": {"encoding": "linear16", "sample_rate": 16000}}
    await adapter.start()
    await adapter.open_call("call-wideband", options)

    pcm16 = b"\x01\x00" * 160
    task = asyncio.create_task(
        _collect_async(adapter.synthesize("call-wideband", "Wideband hello", options))
    )
    await asyncio.sleep(0)
    mock_ws.push(json.dumps({
        "type": "tts_response",
        "audio_data": base64.b64encode(pcm16).decode("ascii"),
        "encoding": "linear16",
        "sample_rate_hz": 16000,
    }))

    assert await task == [pcm16]
    set_mode = json.loads(mock_ws.sent[0])
    request = json.loads(mock_ws.sent[1])
    assert set_mode["output_encoding"] == "linear16"
    assert set_mode["output_sample_rate_hz"] == 16000
    assert request["output_encoding"] == "linear16"
    assert request["output_sample_rate_hz"] == 16000


@pytest.mark.asyncio
async def test_pipeline_orchestrator_resolves_local_adapters():
    app_config = _build_app_config()
    orchestrator = PipelineOrchestrator(app_config)
    await orchestrator.start()

    resolution = orchestrator.get_pipeline("call-99")
    assert resolution is not None
    assert isinstance(resolution.stt_adapter, LocalSTTAdapter)
    assert isinstance(resolution.llm_adapter, LocalLLMAdapter)
    assert isinstance(resolution.tts_adapter, LocalTTSAdapter)
    assert resolution.pipeline_name == "local_only"

    await orchestrator.stop()


# --- whole utterances cut by the engine's VAD (``stt_utterance``) ------------------


@pytest.mark.asyncio
async def test_local_stt_adapter_sends_a_whole_utterance_and_learns_the_servers_capability(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})
    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)
    mock_ws.push(json.dumps({"type": "mode_ready", "mode": "stt", "call_id": "call-utt", "stt_utterances": True}))

    await adapter.start_stream("call-utt", {"mode": "stt", "utterances": True}, sample_rate_hz=16000, fmt="pcm16_16k")
    try:
        assert json.loads(mock_ws.sent[0])["stt_segmenter"] == "client"
        assert adapter.supports_utterances is True
        assert adapter.utterances_supported("call-utt") is True

        audio = b"\x01\x02" * 1600  # 100 ms
        await adapter.send_utterance("call-utt", audio, sample_rate_hz=16000, utterance_id="call-utt:utt-1", fmt="pcm16_16k")

        message = json.loads(mock_ws.sent[-1])
        assert message["type"] == "stt_utterance"
        assert message["mode"] == "stt" and message["call_id"] == "call-utt"
        assert message["rate"] == 16000 and message["format"] == "pcm16le"
        assert message["utterance_id"] == "call-utt:utt-1"
        assert message["duration_ms"] == 100
        assert base64.b64decode(message["data"]) == audio
    finally:
        await adapter.close_call("call-utt")


@pytest.mark.asyncio
async def test_local_stt_adapter_takes_an_unsupported_answer_as_the_fallback_signal(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalSTTAdapter("local_stt", app_config, provider_config, {"mode": "stt"})
    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)

    await adapter.start_stream("call-old", {"mode": "stt"}, sample_rate_hz=16000, fmt="pcm16_16k")
    try:
        assert adapter.utterances_supported("call-old") is None  # an old server never says
        mock_ws.push(json.dumps({
            "type": "stt_result", "text": "", "call_id": "call-old", "mode": "stt",
            "is_final": True, "is_partial": False, "error": "utterances_unsupported", "stt_utterances": False,
        }))
        mock_ws.push(json.dumps({
            "type": "stt_result", "text": "алло", "call_id": "call-old", "mode": "stt",
            "is_final": True, "is_partial": False, "utterance_id": "x",
        }))
        results = adapter.iter_results("call-old")
        assert await asyncio.wait_for(results.__anext__(), timeout=2) == "алло"  # the error result is not a transcript
        assert adapter.utterances_supported("call-old") is False
    finally:
        await adapter.close_call("call-old")


@pytest.mark.asyncio
async def test_local_llm_adapter_skips_the_answer_to_a_cancelled_request(monkeypatch):
    app_config = _build_app_config()
    provider_config = LocalProviderConfig(**app_config.providers["local"])
    adapter = LocalLLMAdapter("local_llm", app_config, provider_config, {"mode": "llm"})
    mock_ws = _MockWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return mock_ws

    monkeypatch.setattr("src.pipelines.local.websockets.connect", fake_connect)
    await adapter.start()
    await adapter.open_call("call-cancel", {"mode": "llm"})

    first = asyncio.create_task(adapter.generate("call-cancel", "у меня три", {"messages": []}, {}))
    await asyncio.sleep(0)
    first_request = json.loads(mock_ws.sent[-1])
    assert first_request["type"] == "llm_request" and first_request["request_id"].startswith("llm-")
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await adapter.cancel_generation("call-cancel")
    cancel = json.loads(mock_ws.sent[-1])
    assert cancel["type"] == "barge_in" and cancel["reason"] == "superseded" and cancel["rollback_assistant"] is False

    second = asyncio.create_task(adapter.generate("call-cancel", "у меня три комнаты", {"messages": []}, {}))
    await asyncio.sleep(0)
    second_request = json.loads(mock_ws.sent[-1])
    # The late answer to the cancelled request, then the acknowledgement, then ours.
    mock_ws.push(json.dumps({"type": "llm_response", "text": "stale", "request_id": first_request["request_id"]}))
    mock_ws.push(json.dumps({"type": "barge_in_ack", "status": "ok", "call_id": "call-cancel"}))
    mock_ws.push(json.dumps({"type": "llm_response", "text": "fresh", "request_id": second_request["request_id"]}))
    assert (await second).text == "fresh"
