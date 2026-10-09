"""Smart Turn judges whether a quiet caller is done or only pausing.

On every Silero stop the engine scores the caller's turn audio with Smart
Turn. A complete verdict changes nothing; an incomplete one holds the turn a
while longer so a caller who is thinking is not answered mid-thought, and a
continuation is judged again on the whole turn.
"""

import asyncio
import time

import pytest

from tests.test_pipeline_end_of_turn_silero import (
    CHUNK,
    GRACE,
    _ScriptedModel,
    _hear,
    _no_turn_within,
)


class _ScriptedTurnModel:
    """Answers with scripted completion probabilities; records what it saw."""

    def __init__(self, probabilities=(), delay_sec=0.0):
        self.probabilities = list(probabilities)
        self.delay_sec = delay_sec
        self.seen = []

    def predict(self, pcm, sample_rate):
        if self.delay_sec:
            time.sleep(self.delay_sec)
        self.seen.append((len(pcm), sample_rate))
        probability = self.probabilities.pop(0) if self.probabilities else 0.9
        return {"probability": probability, "audio_ms": len(pcm) / 2 / sample_rate * 1000, "inference_ms": 1.0}


async def _start_with_turn_model(monkeypatch, call_id, turn_model, llm_options=GRACE, vad=None):
    """Build the engine with Smart Turn loaded before the runner starts."""
    from tests.test_pipeline_end_of_turn_silero import _config
    from unittest.mock import AsyncMock
    from src.core.models import CallSession
    from src.engine import Engine
    from tests.test_pipeline_runner_lifecycle import _RecordingLLM, _ResultStreamingStubSTT, _StubResolution

    silero = _ScriptedModel()
    settings = {"smart_turn_enabled": True, "smart_turn_timeout_ms": 300, **(vad or {})}
    engine = Engine(_config(settings))
    engine.pipeline_orchestrator._started = True
    engine._silero_model = silero
    engine._smart_turn_model = turn_model
    stt = _ResultStreamingStubSTT()
    llm = _RecordingLLM()
    resolution = _StubResolution(stt_adapter=stt, stt_options={"streaming": True, "chunk_ms": 80}, llm_adapter=llm)
    resolution.llm_options = dict(llm_options)
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    session.conversation_state = "listening"
    session.audio_capture_enabled = True
    session.media_rx_confirmed = True
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, session, stt, llm, silero


@pytest.mark.asyncio
async def test_a_complete_verdict_lets_the_turn_go(monkeypatch):
    turn = _ScriptedTurnModel([0.93])
    engine, session, stt, llm, silero = await _start_with_turn_model(monkeypatch, "call-st-complete", turn)
    try:
        assert engine._smart_turn_active("call-st-complete")
        await _hear(engine, session, silero, [0.9, 0.9, 0.9])
        await stt.results.put("не планирую")
        await _hear(engine, session, silero, [0.1, 0.1, 0.1])
        started = time.monotonic()
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert time.monotonic() - started < 1.0
        assert llm.transcripts == ["не планирую"]
        assert turn.seen and turn.seen[0][1] == 16000
        # The verdict was consumed with the turn and the audio starts afresh.
        assert "call-st-complete" not in engine._pipeline_turn_verdict
        assert engine._turn_audio["call-st-complete"].duration_ms == 0
    finally:
        await engine._cleanup_call("call-st-complete")


@pytest.mark.asyncio
async def test_an_incomplete_verdict_holds_the_turn_until_the_hold_expires(monkeypatch):
    turn = _ScriptedTurnModel([0.08])
    engine, session, stt, llm, silero = await _start_with_turn_model(
        monkeypatch, "call-st-hold", turn, vad={"smart_turn_incomplete_hold_ms": 700}
    )
    try:
        await _hear(engine, session, silero, [0.9, 0.9, 0.9])
        await stt.results.put("я хочу")
        await _hear(engine, session, silero, [0.1, 0.1, 0.1])
        stopped = time.monotonic()
        # Past the grace, the turn is still held.
        assert await _no_turn_within(llm, 0.4)
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert time.monotonic() - stopped >= 0.6
        assert llm.transcripts == ["я хочу"]
    finally:
        await engine._cleanup_call("call-st-hold")


@pytest.mark.asyncio
async def test_a_continuation_is_judged_on_the_whole_turn_and_merged(monkeypatch):
    turn = _ScriptedTurnModel([0.05, 0.95])
    engine, session, stt, llm, silero = await _start_with_turn_model(
        monkeypatch, "call-st-continue", turn, vad={"smart_turn_incomplete_hold_ms": 3000}
    )
    try:
        await _hear(engine, session, silero, [0.9, 0.9, 0.9])
        await stt.results.put("я хочу")
        await _hear(engine, session, silero, [0.1, 0.1, 0.1])  # incomplete: held
        await asyncio.sleep(0.3)
        assert not llm.called.is_set()

        await _hear(engine, session, silero, [0.9, 0.9, 0.9])  # the caller goes on
        assert "call-st-continue" not in engine._pipeline_turn_verdict
        await stt.results.put("сделать ремонт")
        await _hear(engine, session, silero, [0.1, 0.1, 0.1])  # complete
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["я хочу сделать ремонт"]
        # The second judgement saw the whole turn, not only the continuation.
        assert len(turn.seen) == 2
        assert turn.seen[1][0] > turn.seen[0][0]
    finally:
        await engine._cleanup_call("call-st-continue")


@pytest.mark.asyncio
async def test_a_slow_verdict_holds_the_turn_only_up_to_the_timeout(monkeypatch):
    turn = _ScriptedTurnModel([0.05], delay_sec=1.5)
    engine, session, stt, llm, silero = await _start_with_turn_model(
        monkeypatch,
        "call-st-slow",
        turn,
        vad={"smart_turn_timeout_ms": 300, "smart_turn_incomplete_hold_ms": 5000},
    )
    try:
        await _hear(engine, session, silero, [0.9, 0.9, 0.9])
        await stt.results.put("алло")
        await _hear(engine, session, silero, [0.1, 0.1, 0.1])
        stopped = time.monotonic()
        await asyncio.wait_for(llm.called.wait(), timeout=3)
        # Released on the timeout, long before the late incomplete verdict.
        assert time.monotonic() - stopped < 1.2
        assert llm.transcripts == ["алло"]
        await asyncio.sleep(1.6)
        # The late verdict found no pending analysis and was dropped.
        assert "call-st-slow" not in engine._pipeline_turn_verdict
    finally:
        await engine._cleanup_call("call-st-slow")


@pytest.mark.asyncio
async def test_the_model_sees_the_turn_minus_the_trailing_silence(monkeypatch):
    turn = _ScriptedTurnModel([0.9])
    engine, session, stt, llm, silero = await _start_with_turn_model(
        monkeypatch, "call-st-trim", turn, vad={"smart_turn_trailing_silence_ms": 0}
    )
    try:
        await _hear(engine, session, silero, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        for _ in range(20):
            if turn.seen:
                break
            await asyncio.sleep(0.02)
        # Six chunks fed, the 96 ms Silero stop trimmed off: three chunks remain.
        assert turn.seen == [(3 * len(CHUNK), 16000)]
    finally:
        await engine._cleanup_call("call-st-trim")


@pytest.mark.asyncio
async def test_pinned_talk_detect_leaves_smart_turn_idle(monkeypatch):
    turn = _ScriptedTurnModel([0.1])
    engine, session, stt, llm, silero = await _start_with_turn_model(
        monkeypatch, "call-st-pinned", turn, llm_options={**GRACE, "end_of_turn_source": "talk_detect"}
    )
    try:
        await _hear(engine, session, silero, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        await asyncio.sleep(0.2)
        assert turn.seen == []
    finally:
        await engine._cleanup_call("call-st-pinned")


@pytest.mark.asyncio
async def test_smart_turn_stays_off_without_silero(monkeypatch):
    from tests.test_pipeline_end_of_turn_silero import _config
    from src.engine import Engine

    engine = Engine(_config({"smart_turn_enabled": True}))
    engine._silero_model = None
    await engine._init_smart_turn()
    assert engine._smart_turn_model is None


@pytest.mark.asyncio
async def test_turn_state_goes_with_the_call(monkeypatch):
    turn = _ScriptedTurnModel([0.1])
    engine, session, stt, llm, silero = await _start_with_turn_model(monkeypatch, "call-st-life", turn)
    await _hear(engine, session, silero, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    await asyncio.sleep(0.1)
    await engine._cleanup_call("call-st-life")
    assert "call-st-life" not in engine._turn_audio
    assert "call-st-life" not in engine._pipeline_turn_verdict
    assert "call-st-life" not in engine._pipeline_turn_verdict_pending
