"""The caller's last words survive a hangup (streaming.pipeline_hangup_final_wait_ms).

A VAD-gated recognizer (GigaAM v3, Sherpa offline) returns a phrase only after
its closing silence, so a caller who answered and hung up at once left their
words in the recognizer: the cleanup tore the pipeline down, the result never
came, and the call record ended with the agent's question. The cleanup now
feeds the recognizer its closing silence, waits a bounded time for the result
and records it as the caller's last turn, with no LLM reply; results the dialog
worker was still holding for the end of the turn are recorded the same way.
"""

import asyncio
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent, TTSComponent
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _StubResolution

LAST_WORDS = "да, перезвоните завтра"


def _new_call_id() -> str:
    # The engine's cleanup guard remembers finished call ids for a while, so
    # every test hangs up a call of its own.
    return f"call-hangup-{uuid4().hex[:8]}"


def _config(*, wait_ms: int = 1500) -> AppConfig:
    return AppConfig(
        **{
            "default_provider": "local",
            "providers": {"local": {"enabled": True}},
            "asterisk": {
                "host": "127.0.0.1",
                "port": 8088,
                "username": "u",
                "password": "p",
                "app_name": "ai-voice-agent",
            },
            "llm": {"initial_greeting": "", "prompt": "You are helpful", "model": "gpt-4o"},
            "pipelines": {"serial": {}},
            "active_pipeline": "serial",
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {
                "pipeline_streaming_overlap": False,
                "pipeline_hangup_final_wait_ms": wait_ms,
            },
        }
    )


class _CountingLLM(LLMComponent):
    supports_streaming = False

    def __init__(self):
        self.calls = []

    async def generate(self, call_id, transcript, context, options):
        self.calls.append(transcript)
        return "ответ"


class _SilentTTS(TTSComponent):
    downstream_mode_override = "stream"

    async def synthesize(self, call_id, text, options):
        yield b"\x00" * 320


class _PlaybackStub:
    def __init__(self):
        self.active = True

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        return "stream-1"

    def is_stream_active(self, call_id, stream_id=None):
        return self.active

    def get_playback_position_ms(self, call_id):
        return 0

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.active = False


async def _start(monkeypatch, *, wait_ms=1500, end_of_turn_silence_ms=100):
    call_id = _new_call_id()
    engine = Engine(_config(wait_ms=wait_ms))
    engine.pipeline_orchestrator._started = True
    monkeypatch.setattr(engine, "streaming_playback_manager", _PlaybackStub())
    stt = _ResultStreamingStubSTT()
    llm = _CountingLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=_SilentTTS(),
    )
    resolution.llm_options = {"end_of_turn_silence_ms": end_of_turn_silence_ms}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "serial"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, session, stt, llm


def _users(session):
    return [m["content"] for m in (session.conversation_history or []) if m.get("role") == "user"]


@pytest.mark.asyncio
async def test_the_result_that_arrives_after_the_hangup_becomes_the_callers_last_turn(monkeypatch):
    engine, session, stt, llm = await _start(monkeypatch)
    # The detector saw the caller stop and asked the recognizer to finalize; the
    # result is still on its way when the caller hangs up.
    engine._pipeline_stt_final_expected_at[session.call_id] = time.monotonic()

    async def late_result():
        await asyncio.sleep(0.2)
        await stt.results.put(LAST_WORDS)

    asyncio.create_task(late_result())
    started = time.monotonic()
    await engine._cleanup_call(session.call_id)
    elapsed = time.monotonic() - started

    assert _users(session) == [LAST_WORDS]
    assert llm.calls == []  # nobody is left to answer
    assert elapsed < 1.2  # the wait ended with the result, not at the limit


@pytest.mark.asyncio
async def test_words_the_worker_was_holding_for_the_end_of_the_turn_are_recorded(monkeypatch):
    engine, session, stt, llm = await _start(monkeypatch, wait_ms=200)
    # Talk detection reports the caller still talking: the worker holds the
    # results it has for the end of the turn instead of answering them.
    engine._note_pipeline_caller_talking(session.call_id, True, source="talk_detect")
    await stt.results.put("подождите")
    await stt.results.put(LAST_WORDS)
    await asyncio.sleep(0.3)
    assert _users(session) == [] and llm.calls == []

    await engine._cleanup_call(session.call_id)

    assert _users(session) == ["подождите " + LAST_WORDS]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_a_quiet_caller_does_not_delay_the_cleanup(monkeypatch):
    engine, session, stt, llm = await _start(monkeypatch)

    started = time.monotonic()
    await engine._cleanup_call(session.call_id)

    assert time.monotonic() - started < 1.0
    assert _users(session) == [] and llm.calls == []


@pytest.mark.asyncio
async def test_zero_turns_the_wait_off(monkeypatch):
    engine, session, stt, llm = await _start(monkeypatch, wait_ms=0)
    engine._pipeline_stt_final_expected_at[session.call_id] = time.monotonic()

    async def late_result():
        await asyncio.sleep(0.3)
        await stt.results.put(LAST_WORDS)

    task = asyncio.create_task(late_result())
    await engine._cleanup_call(session.call_id)
    await task

    assert _users(session) == [] and llm.calls == []


@pytest.mark.asyncio
async def test_a_caller_still_talking_at_hangup_gets_the_recognizer_finalized_and_waited_for(monkeypatch):
    """No finalize was pending: the cleanup feeds the closing silence itself and waits the limit."""
    engine, session, stt, llm = await _start(monkeypatch, wait_ms=300)
    engine._pipeline_caller_talking[session.call_id] = True
    engine._pipeline_caller_talk_changed_at[session.call_id] = time.monotonic()

    started = time.monotonic()
    await engine._cleanup_call(session.call_id)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.3
    silence = sum(len(audio) for audio, _ in stt.sent if audio.count(0) == len(audio))
    assert silence >= 16_000 * 2 * 1.2  # at least 1.2 s of silence reached the recognizer
    assert _users(session) == [] and llm.calls == []


@pytest.mark.asyncio
async def test_a_turn_cancelled_in_the_llm_keeps_the_callers_words(monkeypatch):
    """The caller answered, the LLM was still generating when they hung up."""

    class _BlockingLLM(LLMComponent):
        supports_streaming = False

        def __init__(self):
            self.started = asyncio.Event()

        async def generate(self, call_id, transcript, context, options):
            self.started.set()
            await asyncio.Event().wait()

    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    monkeypatch.setattr(engine, "streaming_playback_manager", _PlaybackStub())
    stt = _ResultStreamingStubSTT()
    llm = _BlockingLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=_SilentTTS(),
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 50}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    call_id = _new_call_id()
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "serial"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)

    await stt.results.put(LAST_WORDS)
    await asyncio.wait_for(llm.started.wait(), timeout=2)
    await engine._cleanup_call(session.call_id)

    assert _users(session) == [LAST_WORDS]
    assert not any(m.get("role") == "assistant" for m in session.conversation_history)
