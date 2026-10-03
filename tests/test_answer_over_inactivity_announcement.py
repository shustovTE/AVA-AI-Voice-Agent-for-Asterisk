"""What the caller says over the watchdog's check-in or final message is heard in full.

The barge-in protection window covers almost all of a short check-in, the
caller's frames are muted for the recognizer while the agent is audible, and a
deferred barge-in is dropped at Silero's stop: a "yes" said entirely over "are
you still there?" reached the recognizer as silence and was dropped, and with
caller sound no longer restarting the check-ins the caller then heard the final
message. Speech over the watchdog's own announcements is now sent whole and
becomes an ordinary turn, taken as the recognizer returns it; and the
watchdog's decision after an announcement waits for an answer still on its way.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.core.utterances import UtteranceCutter
from tests.test_pipeline_end_of_turn_silero import GRACE, _ScriptedModel, _hear, _start_call
from tests.test_silero_protection_ends_with_reply import _capture_utterances, _reply_ended, _reply_playing

VAD = {"silero_stt_utterances": True}
CHECK_IN = "Вы ещё здесь?"


def _announcing(session, text=CHECK_IN, kind="check_in"):
    session.no_input_state = {
        **dict(getattr(session, "no_input_state", None) or {}),
        "announcement_active": True,
        "announcement_text": text,
        "announcement_kind": kind,
    }


def _announced(session):
    session.no_input_state["announcement_active"] = False


@pytest.mark.asyncio
async def test_a_short_answer_said_entirely_over_a_check_in_is_sent_whole(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-whole", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        sent = _capture_utterances(engine)
        _announcing(session)
        await _reply_playing(session, started_ago=0.4)              # the check-in plays, inside the window
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])  # "да"
        await _hear(engine, session, model, [0.1, 0.1, 0.1, 0.1])        # done, while the check-in still plays

        [utterance] = sent
        assert utterance.signal_ms == utterance.duration_ms         # whole: the muted frames are the caller's audio
        assert utterance.interrupted_agent is False                 # nothing was cut; no continuation logic
        engine._apply_barge_in_action.assert_not_awaited()          # the window still protects the check-in
        assert "call-ann-whole" not in engine._pipeline_speech_over_announcement  # done with this utterance
    finally:
        await engine._cleanup_call("call-ann-whole")


@pytest.mark.asyncio
async def test_the_same_speech_over_a_reply_stays_muted(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-reply", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        sent = _capture_utterances(engine)
        await _reply_playing(session, started_ago=0.4)              # an ordinary reply, not an announcement
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])
        await _hear(engine, session, model, [0.1, 0.1, 0.1, 0.1])

        [utterance] = sent
        assert utterance.signal_ms == 0                             # muted, as before: a reply keeps its protection
    finally:
        await engine._cleanup_call("call-ann-reply")


async def _answer_over_check_in(engine, session, model, stt, heard: str):
    """The caller speaks over a check-in that then ends; the recognizer returns ``heard``."""
    _announcing(session)
    await _reply_playing(session, started_ago=0.4)
    await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])
    await _hear(engine, session, model, [0.1, 0.1, 0.1, 0.1])
    assert len(engine._pipeline_utterance_starts[session.call_id]) == 1   # sent to the recognizer
    await _reply_ended(session)
    _announced(session)
    await stt.results.put(heard)


@pytest.mark.asyncio
async def test_an_answer_over_a_check_in_becomes_a_turn(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-turn", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        await _answer_over_check_in(engine, session, model, stt, "да, я здесь")
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["да, я здесь"]
    finally:
        await engine._cleanup_call("call-ann-turn")


@pytest.mark.asyncio
async def test_whatever_is_heard_over_a_check_in_is_taken_as_said(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-native", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        # No filtering of any kind: even words that repeat the check-in are the caller's turn.
        await _answer_over_check_in(engine, session, model, stt, "вы ещё здесь")
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["вы ещё здесь"]
    finally:
        await engine._cleanup_call("call-ann-native")


@pytest.mark.asyncio
async def test_the_announcement_waits_for_an_answer_still_on_its_way(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-wait", GRACE, model, vad=VAD)
    try:
        engine._apply_barge_in_action = AsyncMock()
        await _hear(engine, session, model, [0.9, 0.9, 0.9])          # the caller is talking as the final message ends
        waiting = asyncio.create_task(engine._await_answer_over_announcement("call-ann-wait", kind="final"))
        await asyncio.sleep(0.1)
        assert not waiting.done()                                   # held while Silero hears them
        await _hear(engine, session, model, [0.1, 0.1, 0.1, 0.1])     # they stop: the utterance goes out
        await asyncio.sleep(0.1)
        assert not waiting.done()                                   # held while the result is on its way
        await stt.results.put("подождите, я здесь")
        await asyncio.wait_for(waiting, timeout=1)                  # released once it has arrived
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["подождите, я здесь"]
    finally:
        await engine._cleanup_call("call-ann-wait")


@pytest.mark.asyncio
async def test_nothing_in_progress_means_no_wait(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-nowait", GRACE, model, vad=VAD)
    try:
        started = time.monotonic()
        await engine._await_answer_over_announcement("call-ann-nowait", kind="check_in")
        assert time.monotonic() - started < 0.05
    finally:
        await engine._cleanup_call("call-ann-nowait")


@pytest.mark.asyncio
async def test_a_spoken_check_in_waits_for_the_answer_before_returning(monkeypatch):
    model = _ScriptedModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-ann-wired", GRACE, model, vad=VAD)
    try:
        engine._stream_pipeline_tts_text = AsyncMock(return_value="stream-1")
        engine._await_answer_over_announcement = AsyncMock()
        assert await engine._speak_no_input_announcement("call-ann-wired", CHECK_IN, "check_in") is True
        engine._await_answer_over_announcement.assert_awaited_once_with("call-ann-wired", kind="check_in")
        assert session.no_input_state["announcement_active"] is False
    finally:
        await engine._cleanup_call("call-ann-wired")


# --- the pieces -------------------------------------------------------------------


def _frame(ms, byte):
    return bytes([byte]) * (16 * ms * 2)


def test_a_kept_whole_utterance_reads_its_muted_frames_as_audio_but_interrupted_nothing():
    cutter = UtteranceCutter(sample_rate=16000, preroll_ms=0, max_ms=2000, keep_ms=5000)
    cutter.speech_started()
    cutter.keep_whole()
    for _ in range(5):
        cutter.append(_frame(20, 0x11), muted=True)   # all of it over the announcement
    utterance = cutter.speech_stopped()

    assert utterance.signal_ms == utterance.duration_ms == 100
    assert utterance.pcm16[:2] == b"\x11\x11"
    assert utterance.interrupted_agent is False

