"""The barge-in protection window ends with the reply, never after it.

``talk_detect_initial_protection_ms`` is counted from the start of the agent's
audio. On a short reply ("Hello") the caller's answer that started over its
tail was deferred until the window ran out, the reply ended on its own long
before that, and nothing released the deferral: the frames muted while the
agent was audible stayed silence and the caller's word reached the recognizer
clipped. Now the reply ending on its own ends the window: the utterance is the
caller's from its first frame and is sent whole, without counting as a barge-in.
"""

import time
from unittest.mock import AsyncMock

import pytest

from src.core.utterances import UtteranceCutter
from tests.test_pipeline_end_of_turn_silero import GRACE, _ScriptedModel, _hear, _start_call

VAD = {"silero_stt_utterances": True}


def _capture_utterances(engine):
    sent = []

    def send(call_id, utterance, *, expect_result):
        sent.append(utterance)
        return True

    engine._send_pipeline_utterance = send
    return sent


async def _reply_playing(session, *, started_ago: float):
    session.audio_capture_enabled = False
    session.tts_playing = True
    session.tts_started_ts = time.time() - started_ago


async def _reply_ended(session):
    session.audio_capture_enabled = True
    session.tts_playing = False
    session.tts_ended_ts = time.time()


@pytest.mark.asyncio
async def test_a_reply_that_ends_inside_the_window_releases_the_callers_speech_whole(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-release", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        sent = _capture_utterances(engine)
        assert isinstance(engine._utterance_cutters.get("call-sv-release"), UtteranceCutter)

        await _reply_playing(session, started_ago=0.4)          # a short reply, 400 ms in
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])  # the caller answers over its tail
        assert "call-sv-release" in engine._silero_deferred_barge_in

        await _reply_ended(session)                              # the reply ends on its own, window still open
        await _hear(engine, session, model, [0.9, 0.9, 0.9])     # the caller is still talking
        assert "call-sv-release" not in engine._silero_deferred_barge_in
        engine._apply_barge_in_action.assert_not_awaited()       # nothing was cut

        await _hear(engine, session, model, [0.1, 0.1, 0.1, 0.1])  # they stop
        [utterance] = sent
        assert utterance.signal_ms == utterance.duration_ms      # whole: the muted head is the caller's audio
        assert utterance.interrupted_agent is False              # the reply completed; no continuation logic
    finally:
        await engine._cleanup_call("call-sv-release")


@pytest.mark.asyncio
async def test_the_previous_behaviour_stays_available(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-keep", GRACE, model, vad=VAD)
    try:
        engine.config.barge_in.protection_ends_with_reply = False
        engine._apply_barge_in_action = AsyncMock()
        sent = _capture_utterances(engine)

        await _reply_playing(session, started_ago=0.4)
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])
        await _reply_ended(session)
        await _hear(engine, session, model, [0.9, 0.9, 0.9])
        assert "call-sv-keep" not in engine._silero_deferred_barge_in
        await _hear(engine, session, model, [0.1, 0.1, 0.1, 0.1])

        [utterance] = sent
        assert 0 < utterance.signal_ms < utterance.duration_ms   # the head stays muted, as before
        assert utterance.interrupted_agent is False
    finally:
        await engine._cleanup_call("call-sv-keep")


@pytest.mark.asyncio
async def test_speech_that_outlasts_a_longer_reply_still_interrupts_it_when_the_window_ends(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sv-long", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        await _reply_playing(session, started_ago=0.7)
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])
        assert "call-sv-long" in engine._silero_deferred_barge_in

        session.tts_started_ts = time.time() - 1.6               # the window passed; the reply still plays
        await _hear(engine, session, model, [0.9])
        engine._apply_barge_in_action.assert_awaited_once()
        assert "call-sv-long" not in engine._silero_deferred_barge_in
    finally:
        await engine._cleanup_call("call-sv-long")


# --- the cutter -------------------------------------------------------------------


def _frame(ms, byte):
    return bytes([byte]) * (16 * ms * 2)


def test_an_utterance_the_reply_ended_over_is_returned_whole_but_not_as_an_interruption():
    cutter = UtteranceCutter(sample_rate=16000, preroll_ms=0, max_ms=2000, keep_ms=5000)
    cutter.speech_started()
    for _ in range(5):
        cutter.append(_frame(20, 0x11), muted=True)   # 100 ms over the reply's tail
    cutter.note_reply_ended_over_speech()
    for _ in range(5):
        cutter.append(_frame(20, 0x22), muted=False)  # 100 ms after it
    utterance = cutter.speech_stopped()

    assert utterance.duration_ms == 200
    assert utterance.signal_ms == 200
    assert utterance.pcm16[:2] == b"\x11\x11"          # the muted head is the caller's audio
    assert utterance.interrupted_agent is False

    # The marker is consumed: the next utterance is an ordinary one.
    cutter.speech_started()
    for _ in range(3):
        cutter.append(_frame(20, 0x33), muted=True)
    following = cutter.speech_stopped()
    assert following.signal_ms == 0


def test_a_stale_reply_end_marker_does_not_leak_into_a_later_utterance():
    cutter = UtteranceCutter(sample_rate=16000, preroll_ms=0, max_ms=2000, keep_ms=5000)
    cutter.note_reply_ended_over_speech()             # no utterance was open
    for _ in range(3):
        cutter.append(_frame(20, 0x44), muted=False)
    cutter.speech_started()
    for _ in range(3):
        cutter.append(_frame(20, 0x55), muted=True)
    utterance = cutter.speech_stopped()
    assert utterance.signal_ms == 0
