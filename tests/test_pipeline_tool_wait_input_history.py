"""Caller input stays live while a pipeline reply waits to execute its tools."""

import asyncio
import time
from unittest.mock import AsyncMock
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.core.no_input_watchdog import NoInputPolicy
from src.core.transport_orchestrator import ContextConfig
from src.core.utterances import SttUtterance
from src.engine import Engine
from src.pipelines.base import LLMComponent, LLMResponse, TTSComponent
from src.tools.base import Tool, ToolCategory, ToolDefinition
from src.tools.registry import ToolRegistry
from tests.test_pipeline_runner_lifecycle import (
    _BlockingLLM,
    _ResultStreamingStubSTT,
    _StubResolution,
)
from tests.test_pipeline_turn_preempts_playing_reply import _config


FIRST = "Абонент сейчас не может ответить на ваш звонок."
LAST = "Спасибо. Всего доброго"
CHECK_IN = "Алло. Вы тут?"
REPLY = "Здравствуйте! Это компания ДомЕо, звоню с целью рассчитать стоимость ремонта. Пожалуйста, перезвоните позднее."


class _ProbeTool(Tool):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def definition(self):
        return ToolDefinition(name="probe", description="Test tool", category=ToolCategory.BUSINESS)

    async def execute(self, parameters, context):
        self.started.set()
        await self.release.wait()
        return {"status": "success", "message": "Done"}


class _ToolLLM(LLMComponent):
    supports_streaming = True

    def __init__(self, reply=REPLY):
        self.reply = reply
        self.calls = []
        self._pending_tool_calls_by_call = {}

    @staticmethod
    def _tools():
        return [{"id": "probe-1", "name": "probe", "parameters": {}}]

    async def generate(self, call_id, transcript, context, options):
        self.calls.append(transcript)
        if len(self.calls) == 1:
            return LLMResponse(text=self.reply, tool_calls=self._tools())
        return LLMResponse(text="", tool_calls=[])

    async def generate_stream(self, call_id, transcript, context, options):
        self.calls.append(transcript)
        if len(self.calls) == 1:
            yield self.reply
            self._pending_tool_calls_by_call[call_id] = self._tools()


class _TTS(TTSComponent):
    downstream_mode_override = "stream"

    async def synthesize(self, call_id, text, options):
        yield b"\x00" * 8000


class _Playback:
    """Synthesis ends, but the transport keeps playing until explicitly stopped."""

    def __init__(self):
        self.active_streams = {}
        self.started = asyncio.Event()
        self.position_ms = 500

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.active_streams[call_id] = {"stream_id": "reply-1", **kwargs}
        self.started.set()
        return "reply-1"

    def is_stream_active(self, call_id, stream_id=None):
        info = self.active_streams.get(call_id)
        return bool(info and stream_id in (None, info["stream_id"]))

    def get_playback_position_ms(self, call_id):
        return self.position_ms

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.active_streams.pop(call_id, None)
        return True


async def _until(predicate, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def _start(monkeypatch, *, overlap=False, llm=None):
    data = _config().model_dump()
    data["streaming"]["pipeline_streaming_overlap"] = overlap
    data["contexts"] = {"test": {"prompt": "You are helpful", "tools": ["probe"]}}
    engine = Engine(AppConfig(**data))
    engine.pipeline_orchestrator._started = True
    context = ContextConfig(**data["contexts"]["test"])
    monkeypatch.setattr(
        engine.transport_orchestrator, "get_context_config", lambda *a, **k: context
    )
    playback = _Playback()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    tool = _ProbeTool()
    registry = ToolRegistry.isolated()
    registry._tools["probe"] = tool
    monkeypatch.setattr(engine, "_tool_registry_for_session", lambda _: registry)
    stt = _ResultStreamingStubSTT()
    llm = llm or _ToolLLM()
    resolution = _StubResolution(
        stt_adapter=stt, stt_options={"streaming": True},
        llm_adapter=llm, tts_adapter=_TTS(),
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 10}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    session = CallSession(call_id=f"tool-wait-{uuid4().hex[:8]}", caller_channel_id="caller")
    session.context_name = "test"
    session.pipeline_name = "serial"
    session.media_rx_confirmed = True
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), 2)
    return engine, session, stt, llm, playback, tool


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", [False, True])
async def test_barge_in_releases_the_tool_wait_as_soon_as_playback_stops(monkeypatch, overlap):
    engine, session, stt, llm, playback, tool = await _start(monkeypatch, overlap=overlap)
    try:
        await stt.results.put(FIRST)
        await _until(lambda: (r := engine._spoken_replies.get(session.call_id)) is not None and r.completed)
        assert not tool.started.is_set()
        await engine._apply_barge_in_action(session.call_id, source="silero_vad", reason="test")
        # The old text-length sleep lasted over 8 seconds after synthesis.
        await asyncio.wait_for(tool.started.wait(), 0.5)
        assert not playback.is_stream_active(session.call_id)
        assert engine._spoken_replies[session.call_id].interrupted is True
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_short_reply_does_not_execute_tools_while_audio_is_still_playing(monkeypatch):
    engine, session, stt, llm, playback, tool = await _start(monkeypatch, llm=_ToolLLM("Ок."))
    try:
        await stt.results.put(FIRST)
        await _until(lambda: (r := engine._spoken_replies.get(session.call_id)) is not None and r.completed)
        await asyncio.sleep(0.35)  # longer than the old 3 * 0.08 estimate
        assert not tool.started.is_set()
        await playback.stop_streaming_playback(session.call_id)
        await asyncio.wait_for(tool.started.wait(), 0.5)
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_final_queued_during_a_tool_prevents_a_false_idle_check_in(monkeypatch):
    engine, session, stt, llm, playback, tool = await _start(monkeypatch, llm=_ToolLLM(""))
    announce = AsyncMock(return_value=True)
    engine.no_input_watchdog._announce = announce
    await engine.no_input_watchdog.register(
        session.call_id, NoInputPolicy(initial_timeout_sec=0.05, grace_timeout_sec=0.05),
        is_outbound=False,
    )
    await engine.no_input_watchdog.mark_ready(session.call_id)
    try:
        await stt.results.put(FIRST)
        await asyncio.wait_for(tool.started.wait(), 1)
        previous_final = engine._pipeline_last_final_at[session.call_id]
        await stt.results.put(LAST)
        await _until(lambda: engine._pipeline_last_final_at[session.call_id] > previous_final)
        state = engine.no_input_watchdog.snapshot(session.call_id)
        assert state["last_activity_source"] == "pipeline:stt_final"
        # A playback-end callback used to clear processing even with a result queued.
        await engine.no_input_watchdog.note_agent_output_end(session.call_id)
        await asyncio.sleep(0.15)
        announce.assert_not_awaited()
        assert llm.calls == [FIRST]
        tool.release.set()
        await _until(lambda: LAST in llm.calls)
        await _until(lambda: not engine.no_input_watchdog.snapshot(session.call_id)["pending_input"])
        await _until(lambda: announce.await_count > 0)
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_hangup_places_delayed_words_before_an_announcement_started_after_the_speech(monkeypatch):
    engine, session, stt, llm, playback, tool = await _start(monkeypatch, llm=_BlockingLLM())
    try:
        await stt.results.put(FIRST)
        await asyncio.wait_for(llm.started.wait(), 1)
        spoken_at = time.monotonic()
        speech_wall = time.time()
        assert engine._send_pipeline_utterance(
            session.call_id,
            SttUtterance(
                pcm16=b"\x00" * 16000, sample_rate=16000, utterance_id="last",
                started_at=spoken_at, ended_at=spoken_at + 0.5, signal_ms=500,
            ),
            expect_result=True,
        )
        # An announcement can already have won its scheduling race while STT
        # is still returning the caller's earlier words.
        monkeypatch.setattr(engine, "_stream_pipeline_tts_text", AsyncMock(return_value="check-in"))
        assert await engine._speak_no_input_announcement(session.call_id, CHECK_IN, "check_in")
        previous_final = engine._pipeline_last_final_at[session.call_id]
        await stt.results.put(LAST)
        await _until(lambda: engine._pipeline_last_final_at[session.call_id] > previous_final)
        await engine._cleanup_call(session.call_id)
        history = session.conversation_history
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", FIRST), ("user", LAST), ("assistant", CHECK_IN),
        ]
        assert abs(history[1]["timestamp"] - speech_wall) < 0.05
        assert history[1]["timestamp"] <= history[2]["timestamp"]
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_replacement_stream_does_not_extend_the_previous_replys_tool_wait():
    engine = Engine.__new__(Engine)
    engine.streaming_playback_manager = _Playback()
    session = CallSession(call_id="replaced-tool-stream", caller_channel_id="caller")
    engine.streaming_playback_manager.active_streams[session.call_id] = {"stream_id": "replacement"}

    assert await engine._wait_for_pipeline_tool_playback(session, stream_id="reply-1", timeout_sec=0.03)
    assert engine.streaming_playback_manager.is_stream_active(session.call_id, "replacement")


@pytest.mark.asyncio
async def test_a_stalled_stream_times_out_without_treating_playback_as_complete():
    engine = Engine.__new__(Engine)
    engine.streaming_playback_manager = _Playback()
    session = CallSession(call_id="stalled-tool-stream", caller_channel_id="caller")
    engine.streaming_playback_manager.active_streams[session.call_id] = {"stream_id": "reply-1"}

    assert not await engine._wait_for_pipeline_tool_playback(session, stream_id="reply-1", timeout_sec=0.03)


@pytest.mark.asyncio
async def test_streaming_fallback_waits_for_the_replacement_file_before_executing_tools(monkeypatch):
    class FailOnceTTS(_TTS):
        calls = 0

        async def synthesize(self, call_id, text, options):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("streaming synthesis failed")
            yield b"\x00" * 8000

    engine, session, stt, llm, playback, tool = await _start(monkeypatch)
    engine.pipeline_orchestrator.get_pipeline(session.call_id, session.pipeline_name).tts_adapter = FailOnceTTS()
    waiting, finished = asyncio.Event(), asyncio.Event()

    async def wait_for_file(call_id, playback_id, *, timeout_sec):
        assert playback_id == "file-1"
        waiting.set()
        await finished.wait()
        return True

    file_player = SimpleNamespace(
        play_audio=AsyncMock(return_value="file-1"), wait_for_playback_end=wait_for_file,
    )
    monkeypatch.setattr(engine, "playback_manager", file_player)
    try:
        await stt.results.put(FIRST)
        await asyncio.wait_for(waiting.wait(), 1)
        assert not tool.started.is_set()
        finished.set()
        await asyncio.wait_for(tool.started.wait(), 0.5)
    finally:
        await engine._cleanup_call(session.call_id)
