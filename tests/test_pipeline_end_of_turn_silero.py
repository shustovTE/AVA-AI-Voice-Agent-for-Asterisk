"""Silero VAD as the single source of truth for the end of a caller's turn.

With ``vad.silero_enabled`` the engine scores the caller's own frames, holds
the turn while Silero reports speech, tells the recognizer to finalize the
moment it reports quiet, and releases the turn a short grace later. Asterisk
talk-detect events keep serving barge-in and the watchdog but no longer decide
the turn, so the two detectors cannot disagree about it.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from tests.test_pipeline_runner_lifecycle import (
    _RecordingLLM,
    _ResultStreamingStubSTT,
    _StubResolution,
)

CHUNK = b"\x00" * 1024  # one 32 ms chunk at 16 kHz: 512 samples
GRACE = {"end_of_turn_talk_detect_grace_ms": 150}


class _ScriptedModel:
    """A stand-in for the ONNX graph that answers with scripted probabilities."""

    def __init__(self, probabilities=()):
        self.probabilities = list(probabilities)
        self.rates = []

    def run(self, samples, state, sample_rate):
        self.rates.append(sample_rate)
        probability = self.probabilities.pop(0) if self.probabilities else 0.0
        return probability, state


def _config(vad=None) -> AppConfig:
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
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "externalmedia",
            "vad": {
                "silero_enabled": True,
                "silero_start_ms": 96,
                "silero_stop_ms": 96,
                "silero_stt_finalize_ms": 900,
                **(vad or {}),
            },
        }
    )


async def _start_call(monkeypatch, call_id, llm_options, model, vad=None):
    engine = Engine(_config(vad))
    engine.pipeline_orchestrator._started = True
    engine._silero_model = model
    stt = _ResultStreamingStubSTT()
    llm = _RecordingLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
    )
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
    return engine, session, stt, llm


async def _hear(engine, session, model, probabilities, rate=16000, chunk=CHUNK):
    """Feed one chunk per scripted probability through the engine's hook."""
    model.probabilities.extend(probabilities)
    for _ in probabilities:
        await engine._observe_silero_vad(session, chunk, rate, source="test")


async def _no_turn_within(llm, seconds):
    await asyncio.sleep(seconds)
    return not llm.called.is_set()


def _silence_burst(stt):
    return [audio for audio, _fmt in stt.sent if len(audio) == 28800 and not any(audio)]


@pytest.mark.asyncio
async def test_auto_source_prefers_silero_and_ignores_talk_detect_events(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-auto", GRACE, model)
    try:
        assert engine._pipeline_turn_source["call-sv-auto"] == "vad"

        await _hear(engine, session, model, [0.9, 0.9, 0.9])  # Silero: talking
        await stt.results.put("я хочу")
        # Asterisk reporting quiet is not this call's source of truth any more.
        engine._note_pipeline_caller_talking("call-sv-auto", False)
        assert await _no_turn_within(llm, 0.5)

        await _hear(engine, session, model, [0.1, 0.1, 0.1])  # Silero: quiet
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["я хочу"]
    finally:
        await engine._cleanup_call("call-sv-auto")


@pytest.mark.asyncio
async def test_quiet_feeds_the_recognizer_a_silence_burst_to_finalize(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-burst", GRACE, model)
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        await asyncio.wait_for(stt.audio_sent.wait(), timeout=2)
        for _ in range(20):
            if _silence_burst(stt):
                break
            await asyncio.sleep(0.05)
        # 900 ms of PCM16 at 16 kHz, in one piece, so the recognizer's own
        # gate is satisfied at once instead of in real time.
        assert len(_silence_burst(stt)) == 1
    finally:
        await engine._cleanup_call("call-sv-burst")


@pytest.mark.asyncio
async def test_no_burst_is_sent_when_finalization_is_disabled(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch, "call-sv-noburst", GRACE, model, vad={"silero_stt_finalize_ms": 0}
    )
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        await asyncio.sleep(0.3)
        assert _silence_burst(stt) == []
    finally:
        await engine._cleanup_call("call-sv-noburst")


@pytest.mark.asyncio
async def test_turn_waits_for_the_result_the_recognizer_was_told_to_produce(monkeypatch):
    """Speech after the last result means a result is still on its way."""
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch,
        "call-sv-wait",
        {**GRACE, "end_of_turn_vad_final_wait_ms": 1500},
        model,
    )
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        await stt.results.put("первая часть")
        await asyncio.sleep(0.05)
        await _hear(engine, session, model, [0.9, 0.9])  # more speech after that result
        await _hear(engine, session, model, [0.1, 0.1, 0.1])  # quiet: finalize requested
        assert "call-sv-wait" in engine._pipeline_stt_final_expected_at
        # Well past the grace, the turn is still waiting for the tail.
        assert await _no_turn_within(llm, 0.6)

        await stt.results.put("вторая часть")
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["первая часть вторая часть"]
        assert "call-sv-wait" not in engine._pipeline_stt_final_expected_at
    finally:
        await engine._cleanup_call("call-sv-wait")


@pytest.mark.asyncio
async def test_a_result_that_never_comes_is_bounded_by_the_final_wait(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch,
        "call-sv-bound",
        {**GRACE, "end_of_turn_vad_final_wait_ms": 300},
        model,
    )
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        await stt.results.put("алло")
        await asyncio.sleep(0.05)
        await _hear(engine, session, model, [0.9, 0.9, 0.1, 0.1, 0.1])
        started = time.monotonic()
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["алло"]
        assert time.monotonic() - started < 1.0
    finally:
        await engine._cleanup_call("call-sv-bound")


@pytest.mark.asyncio
async def test_a_result_after_the_last_speech_needs_no_wait(monkeypatch):
    """The recognizer's own gate fired first: nothing is outstanding."""
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch,
        "call-sv-nowait",
        {**GRACE, "end_of_turn_vad_final_wait_ms": 5000},
        model,
    )
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        await stt.results.put("не планирую")
        await asyncio.sleep(0.05)
        await _hear(engine, session, model, [0.1, 0.1, 0.1])
        assert "call-sv-nowait" not in engine._pipeline_stt_final_expected_at
        started = time.monotonic()
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["не планирую"]
        assert time.monotonic() - started < 1.0
    finally:
        await engine._cleanup_call("call-sv-nowait")


@pytest.mark.asyncio
async def test_pinned_talk_detect_leaves_silero_out_of_the_turn(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch,
        "call-sv-pinned",
        {**GRACE, "end_of_turn_source": "talk_detect"},
        model,
    )
    try:
        assert engine._pipeline_turn_source["call-sv-pinned"] == "talk_detect"
        await _hear(engine, session, model, [0.9, 0.9, 0.9])  # Silero talking: ignored
        await stt.results.put("да давай")
        await asyncio.wait_for(llm.called.wait(), timeout=1)
        assert llm.transcripts == ["да давай"]
    finally:
        await engine._cleanup_call("call-sv-pinned")


@pytest.mark.asyncio
async def test_speech_during_playback_barges_in_once_past_the_protection(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-barge", GRACE, model)
    try:
        engine._apply_barge_in_action = AsyncMock()
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time()  # inside talk_detect_initial_protection_ms
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        engine._apply_barge_in_action.assert_not_awaited()

        await _hear(engine, session, model, [0.1, 0.1, 0.1])
        session.tts_started_ts = time.time() - 5.0
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        engine._apply_barge_in_action.assert_awaited_once()
        assert engine._apply_barge_in_action.await_args.kwargs["source"] == "silero_vad"
        assert session.audio_capture_enabled is True
    finally:
        await engine._cleanup_call("call-sv-barge")


@pytest.mark.asyncio
async def test_barge_in_can_be_left_to_asterisk(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch, "call-sv-nobarge", GRACE, model, vad={"silero_barge_in": False}
    )
    try:
        assert engine._silero_owns_barge_in("call-sv-nobarge") is False
        engine._apply_barge_in_action = AsyncMock()
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time() - 5.0
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        engine._apply_barge_in_action.assert_not_awaited()
    finally:
        await engine._cleanup_call("call-sv-nobarge")


@pytest.mark.asyncio
async def test_other_wire_rates_are_resampled_before_scoring(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-rate", GRACE, model)
    try:
        frame = np.full(960, 1000, dtype="<i2").tobytes()  # 20 ms at 48 kHz
        await _hear(engine, session, model, [0.9] * 6, rate=48000, chunk=frame)
        assert model.rates and set(model.rates) == {16000}
    finally:
        await engine._cleanup_call("call-sv-rate")


@pytest.mark.asyncio
async def test_tracker_needs_a_loaded_model_and_goes_with_the_call(monkeypatch):
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-life", GRACE, None)
    try:
        assert "call-sv-life" not in engine._silero_trackers
        assert engine._pipeline_turn_source["call-sv-life"] == "talk_detect"
    finally:
        await engine._cleanup_call("call-sv-life")

    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-life2", GRACE, model)
    assert "call-sv-life2" in engine._silero_trackers
    await engine._cleanup_call("call-sv-life2")
    assert "call-sv-life2" not in engine._silero_trackers
    assert "call-sv-life2" not in engine._pipeline_turn_source
    assert "call-sv-life2" not in engine._pipeline_stt_final_expected_at
