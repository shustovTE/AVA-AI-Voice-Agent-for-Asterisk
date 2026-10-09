"""Barge-in must stop generation even between sentence/TTS boundaries."""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.engine import Engine, _PipelinePlaybackInterrupted
from src.pipelines.base import LLMComponent, TTSComponent
from tests.test_pipeline_unheard_reply_discard import _start, _wait_for


FIRST = "Первое предложение."
SECOND = "Второе предложение."


class _WaitingLLM(LLMComponent):
    supports_streaming = True

    def __init__(self, *, before_first=False, tail=SECOND + " "):
        self.before_first = before_first
        self.tail = tail
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.cancel_requests = []

    async def generate(self, *args):
        raise AssertionError("Interrupted streaming must not retry serial generation")

    async def generate_stream(self, *args):
        try:
            if not self.before_first:
                yield FIRST + " "
            self.waiting.set()
            await self.release.wait()
            yield self.tail
        finally:
            self.closed.set()

    async def cancel_generation(self, call_id):
        self.cancel_requests.append(call_id)


class _TrackingTTS(TTSComponent):
    downstream_mode_override = "stream"

    def __init__(self, *, block=False):
        self.requests = []
        self.block = block
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.engine = None
        self.playback = None

    async def synthesize(self, call_id, text, options):
        self.requests.append(text)
        try:
            self.playback.position_ms = 40
            self.engine._mark_pipeline_reply_audible(call_id)
            yield b"\x00" * 320
            if self.block:
                self.waiting.set()
                await self.release.wait()
                yield b"\x00" * 320
        finally:
            self.closed.set()


async def _start_stream(monkeypatch, llm, *, tts=None, **streaming):
    tts = tts or _TrackingTTS()
    engine, session, stt, playback, tracker = await _start(
        monkeypatch, llm=llm, tts=tts,
        streaming={"pipeline_streaming_overlap": True, **streaming},
    )
    engine.ari_client.hangup_channel = AsyncMock()
    tts.engine, tts.playback = engine, playback
    await stt.results.put("Расскажите о ремонте")
    return engine, session, playback, tts


@pytest.mark.parametrize("heard_reply", [False, True])
async def test_barge_in_cancels_llm_between_sentences(monkeypatch, heard_reply):
    llm = _WaitingLLM()
    engine, session, playback, tts = await _start_stream(
        monkeypatch, llm, pipeline_heard_reply_on_interrupt=heard_reply,
    )
    try:
        await asyncio.wait_for(llm.waiting.wait(), 2)
        await engine._apply_barge_in_action(session.call_id, source="silero_vad", reason="test")
        # Do not release the next token: interruption must close the pending read.
        assert await _wait_for(llm.closed.is_set, timeout=0.3)
        assert llm.cancel_requests == [session.call_id]
        assert tts.requests == [FIRST]
        assert not playback.active
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        history = session.conversation_history
        assert any(m["role"] == "user" and m["content"] == "Расскажите о ремонте" for m in history)
        assert not any(SECOND in m.get("content", "") for m in history)
    finally:
        llm.release.set()
        await engine._cleanup_call(session.call_id)


@pytest.mark.parametrize("tail", [SECOND + " ", "Неоконченное предложение"])
async def test_barge_in_wins_over_ready_llm_text_and_remainder(monkeypatch, tail):
    llm = _WaitingLLM(tail=tail)
    engine, session, playback, tts = await _start_stream(monkeypatch, llm)
    try:
        await asyncio.wait_for(llm.waiting.wait(), 2)
        llm.release.set()
        await engine._apply_barge_in_action(session.call_id, source="silero_vad", reason="test")
        assert await _wait_for(llm.closed.is_set)
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        assert tts.requests == [FIRST]
    finally:
        llm.release.set()
        await engine._cleanup_call(session.call_id)


async def test_barge_in_during_tts_closes_paused_llm(monkeypatch):
    llm = _WaitingLLM()
    tts = _TrackingTTS(block=True)
    engine, session, playback, tts = await _start_stream(monkeypatch, llm, tts=tts)
    try:
        await asyncio.wait_for(tts.waiting.wait(), 2)
        await engine._apply_barge_in_action(session.call_id, source="silero_vad", reason="test")
        assert await _wait_for(tts.closed.is_set, timeout=0.3)
        assert await _wait_for(llm.closed.is_set, timeout=0.3)
        assert llm.cancel_requests == [session.call_id]
        assert tts.requests == [FIRST]
    finally:
        tts.release.set()
        await engine._cleanup_call(session.call_id)


async def test_unheard_overlap_reply_cancels_without_waiting_for_first_token(monkeypatch):
    llm = _WaitingLLM(before_first=True)
    engine, session, playback, tts = await _start_stream(
        monkeypatch, llm, pipeline_discard_unheard_reply=True,
    )
    try:
        await asyncio.wait_for(llm.waiting.wait(), 2)
        # Keep the continued caller turn open so cancellation cannot start a retry.
        engine._note_pipeline_caller_talking(session.call_id, True, source="vad")
        assert await engine._discard_unheard_reply(session)
        engine._pipeline_caller_resumed[session.call_id].set()
        assert await _wait_for(llm.closed.is_set, timeout=0.3)
        assert llm.cancel_requests == [session.call_id]
        assert tts.requests == []
    finally:
        llm.release.set()
        await engine._cleanup_call(session.call_id)


async def test_caller_activity_without_barge_in_does_not_cancel_audible_reply(monkeypatch):
    llm = _WaitingLLM()
    engine, session, playback, tts = await _start_stream(monkeypatch, llm)
    try:
        await asyncio.wait_for(llm.waiting.wait(), 2)
        engine._pipeline_caller_resumed[session.call_id].set()
        await asyncio.sleep(0.02)
        assert not llm.closed.is_set()
        llm.release.set()
        assert await _wait_for(lambda: len(tts.requests) == 2)
        assert await _wait_for(llm.closed.is_set)
        assert tts.requests == [FIRST, SECOND]
        assert llm.cancel_requests == []
    finally:
        llm.release.set()
        await engine._cleanup_call(session.call_id)


async def test_hangup_closes_pending_stream_and_cancels_generation(monkeypatch):
    llm = _WaitingLLM()
    engine, session, playback, tts = await _start_stream(monkeypatch, llm)
    await asyncio.wait_for(llm.waiting.wait(), 2)
    await engine._cleanup_call(session.call_id)
    assert llm.closed.is_set()
    assert llm.cancel_requests == [session.call_id]
    assert tts.requests == [FIRST]


@pytest.mark.parametrize("ready_chunk", [False, True])
async def test_stopped_stream_never_starts_another_synthesis(ready_chunk):
    engine = Engine.__new__(Engine)
    engine._pipeline_reply_inflight = {}
    event = asyncio.Event()
    event.set()
    engine._pipeline_caller_resumed = {"call": event}
    engine.streaming_playback_manager = SimpleNamespace(is_stream_active=lambda *args: False)
    started = False

    async def synthesize():
        nonlocal started
        started = True
        if not ready_chunk:
            await asyncio.Event().wait()
        yield b"audio"

    with pytest.raises(_PipelinePlaybackInterrupted):
        async for _ in engine._pipeline_chunks_while_wanted("call", "old-stream", synthesize()):
            pass
    assert not started


@pytest.mark.parametrize("outcome", ["chunk", "eof", "error"])
async def test_interruption_wins_when_provider_read_also_finishes(outcome):
    engine = Engine.__new__(Engine)
    engine._pipeline_reply_inflight = {}
    event = asyncio.Event()
    engine._pipeline_caller_resumed = {"call": event}
    active = True
    engine.streaming_playback_manager = SimpleNamespace(is_stream_active=lambda *args: active)

    async def chunks():
        nonlocal active
        # Both the wake event and the provider read are ready in the same tick.
        active = False
        event.set()
        if outcome == "error":
            raise RuntimeError("late provider error")
        if outcome == "chunk":
            yield b"stale audio"

    with pytest.raises(_PipelinePlaybackInterrupted):
        async with contextlib.aclosing(engine._pipeline_chunks_while_wanted("call", "stream", chunks())) as stream:
            async for _ in stream:
                pytest.fail("A ready chunk must not beat interruption")


@pytest.mark.parametrize("flag", ["stop_requested", "end_reason"])
async def test_stopping_stream_rejects_new_output_before_cleanup_finishes(flag):
    engine = Engine.__new__(Engine)
    engine._pipeline_reply_inflight = {}
    engine._pipeline_caller_resumed = {}
    info = {"stream_id": "stream", flag: True if flag == "stop_requested" else "barge-in"}
    engine.streaming_playback_manager = SimpleNamespace(
        is_stream_active=lambda *args: True,
        active_streams={"call": info},
    )
    started = False

    async def chunks():
        nonlocal started
        started = True
        yield b"audio"

    with pytest.raises(_PipelinePlaybackInterrupted):
        async for _ in engine._pipeline_chunks_while_wanted("call", "stream", chunks()):
            pass
    assert not started
