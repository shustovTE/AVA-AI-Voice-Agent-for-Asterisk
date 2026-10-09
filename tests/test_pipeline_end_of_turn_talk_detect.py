"""The end of a caller's turn is decided by Asterisk talk detection.

A streaming recognizer only emits a result after its own silence gate, so a
window measured from the result can never bridge a caller who pauses and goes
on: the continuation's result arrives only after they pause again. In a
production call «слушай я пока вообще не знаю» was answered 600 ms after its
result while «сейчас» followed 1.6 s later, into a closed microphone.

Driven by the same TALK_DETECT events that trigger barge-in, the turn is held
while Asterisk reports the caller talking and released a short grace after it
reports them quiet.
"""
import asyncio

import pytest

from src.engine import EndOfTurnPolicy
from tests.test_pipeline_end_of_turn import _start_call

TALK = {"end_of_turn_source": "talk_detect", "end_of_turn_talk_detect_grace_ms": 150}


async def _no_turn_within(llm, seconds):
    await asyncio.sleep(seconds)
    return not llm.called.is_set()


@pytest.mark.asyncio
async def test_a_result_is_held_while_the_caller_keeps_talking(monkeypatch):
    """The production case, replayed: pause, result, continuation, then quiet."""
    engine, stt, llm = await _start_call(monkeypatch, "call-td-hold", TALK)
    try:
        engine._note_pipeline_caller_talking("call-td-hold", True)
        await stt.results.put("слушай я пока вообще не знаю")
        assert await _no_turn_within(llm, 0.5)

        await stt.results.put("сейчас")
        assert await _no_turn_within(llm, 0.5)

        engine._note_pipeline_caller_talking("call-td-hold", False)
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["слушай я пока вообще не знаю сейчас"]
    finally:
        await engine._cleanup_call("call-td-hold")


@pytest.mark.asyncio
async def test_a_result_while_quiet_is_released_after_the_grace(monkeypatch):
    engine, stt, llm = await _start_call(monkeypatch, "call-td-quiet", TALK)
    try:
        engine._note_pipeline_caller_talking("call-td-quiet", False)
        await stt.results.put("не планирую")

        await asyncio.wait_for(llm.called.wait(), timeout=1)
        assert llm.transcripts == ["не планирую"]
    finally:
        await engine._cleanup_call("call-td-quiet")


@pytest.mark.asyncio
async def test_a_result_that_lands_after_quiet_still_joins_the_turn(monkeypatch):
    """The recognizer's gate can outlast Asterisk's silence by a few hundred ms."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-td-late",
        {**TALK, "end_of_turn_talk_detect_grace_ms": 400},
    )
    try:
        engine._note_pipeline_caller_talking("call-td-late", True)
        await stt.results.put("первая часть")
        engine._note_pipeline_caller_talking("call-td-late", False)
        await asyncio.sleep(0.2)
        await stt.results.put("вторая часть")

        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["первая часть вторая часть"]
    finally:
        await engine._cleanup_call("call-td-late")


@pytest.mark.asyncio
async def test_resumed_talking_cancels_a_pending_release(monkeypatch):
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-td-resume",
        {**TALK, "end_of_turn_talk_detect_grace_ms": 400},
    )
    try:
        engine._note_pipeline_caller_talking("call-td-resume", False)
        await stt.results.put("я хочу")
        await asyncio.sleep(0.1)
        engine._note_pipeline_caller_talking("call-td-resume", True)
        assert await _no_turn_within(llm, 0.8)

        await stt.results.put("сделать ремонт")
        engine._note_pipeline_caller_talking("call-td-resume", False)
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["я хочу сделать ремонт"]
    finally:
        await engine._cleanup_call("call-td-resume")


@pytest.mark.asyncio
async def test_a_lost_finished_event_is_bounded_by_the_hold(monkeypatch):
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-td-lost",
        {**TALK, "end_of_turn_talk_detect_hold_ms": 500},
    )
    try:
        engine._note_pipeline_caller_talking("call-td-lost", True)
        await stt.results.put("алло")

        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["алло"]
    finally:
        await engine._cleanup_call("call-td-lost")


@pytest.mark.asyncio
async def test_talking_without_a_result_never_fires(monkeypatch):
    engine, stt, llm = await _start_call(monkeypatch, "call-td-empty", TALK)
    try:
        engine._note_pipeline_caller_talking("call-td-empty", True)
        engine._note_pipeline_caller_talking("call-td-empty", False)
        assert await _no_turn_within(llm, 0.5)
    finally:
        await engine._cleanup_call("call-td-empty")


@pytest.mark.asyncio
async def test_final_source_ignores_talk_detection(monkeypatch):
    """The legacy window can be pinned explicitly."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-td-off",
        {"end_of_turn_source": "final", "end_of_turn_silence_ms": 100},
    )
    try:
        engine._note_pipeline_caller_talking("call-td-off", True)
        await stt.results.put("да давай")

        await asyncio.wait_for(llm.called.wait(), timeout=1)
        assert llm.transcripts == ["да давай"]
    finally:
        await engine._cleanup_call("call-td-off")


@pytest.mark.asyncio
async def test_auto_source_follows_the_pipeline_talk_detect_flag(monkeypatch):
    engine, stt, llm = await _start_call(
        monkeypatch, "call-td-auto", {"end_of_turn_silence_ms": 100}
    )
    try:
        session = await engine.session_store.get_by_call_id("call-td-auto")
        session.vad_state = {"pipeline_talk_detect": {"enabled": True}}
        await engine.session_store.upsert_call(session)

        engine._note_pipeline_caller_talking("call-td-auto", True)
        await stt.results.put("я хочу")
        assert await _no_turn_within(llm, 0.5)

        engine._note_pipeline_caller_talking("call-td-auto", False)
        await asyncio.wait_for(llm.called.wait(), timeout=2)
    finally:
        await engine._cleanup_call("call-td-auto")


# --- policy parsing -----------------------------------------------------------


def test_policy_defaults():
    policy = EndOfTurnPolicy()

    assert policy.source == "auto"
    assert policy.talk_detect_grace_ms == EndOfTurnPolicy.DEFAULT_TALK_DETECT_GRACE_MS
    assert policy.talk_detect_hold_ms == EndOfTurnPolicy.DEFAULT_TALK_DETECT_HOLD_MS
    assert policy.uses_talk_detect(True) is True
    assert policy.uses_talk_detect(False) is False


@pytest.mark.parametrize(
    "value,expected",
    [("talk_detect", "talk_detect"), ("FINAL", "final"), ("auto", "auto"), ("bogus", "auto"), (None, "auto")],
)
def test_policy_source_parsing(value, expected):
    assert EndOfTurnPolicy({"end_of_turn_source": value}).source == expected


def test_policy_explicit_sources_ignore_the_flag():
    assert EndOfTurnPolicy({"end_of_turn_source": "talk_detect"}).uses_talk_detect(False) is True
    assert EndOfTurnPolicy({"end_of_turn_source": "final"}).uses_talk_detect(True) is False


def test_policy_clamps_and_parses_windows():
    policy = EndOfTurnPolicy(
        {"end_of_turn_talk_detect_grace_ms": "300", "end_of_turn_talk_detect_hold_ms": -5}
    )

    assert policy.talk_detect_grace_ms == pytest.approx(300.0)
    assert policy.talk_detect_hold_ms == 0.0
