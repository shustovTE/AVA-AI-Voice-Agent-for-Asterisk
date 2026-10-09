"""Gated caller audio reaches the recognizer as silence, not as a splice.

While the agent speaks the caller's frames are withheld so the agent does not
hear itself. Dropping them outright glues the audio either side of the gap
together, and a word straddling a short gap comes out garbled. A no-input
check-in punches exactly such a gap into the middle of a caller's sentence.
"""
import asyncio

import pytest

from src.engine import PIPELINE_GATED_SILENCE_MS_DEFAULT, PIPELINE_STT_SAMPLE_RATE_HZ, Engine

CALL_ID = "call-gated"
# 20 ms of PCM16 at 8 kHz, as an AudioSocket frame arrives.
FRAME_8K = b"\x11\x22" * 160


def _engine(budget_ms=None, maxsize=1000):
    engine = Engine.__new__(Engine)
    engine._pipeline_queues = {CALL_ID: asyncio.Queue(maxsize=maxsize)}
    engine._resample_state_pipeline16k = {}
    engine._pipeline_gated_silence_ms = {} if budget_ms is None else {CALL_ID: budget_ms}
    engine._pipeline_gated_silence_used_ms = {}
    return engine


def _drain(engine):
    q = engine._pipeline_queues[CALL_ID]
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_a_gated_frame_arrives_as_silence():
    engine = _engine()

    engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)

    frames = _drain(engine)
    assert len(frames) == 1
    assert set(frames[0]) == {0}


def test_the_silence_is_resampled_to_the_pipeline_rate():
    """20 ms stays 20 ms, so the recognizer's timeline does not drift."""
    engine = _engine()

    engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)

    frame = _drain(engine)[0]
    expected_samples = int(PIPELINE_STT_SAMPLE_RATE_HZ * 0.020)
    assert len(frame) == pytest.approx(expected_samples * 2, abs=4)


def test_a_frame_already_at_the_pipeline_rate_is_passed_through():
    engine = _engine()
    frame_16k = b"\x11\x22" * 320

    engine._feed_pipeline_silence(CALL_ID, frame_16k, PIPELINE_STT_SAMPLE_RATE_HZ)

    assert _drain(engine) == [bytes(len(frame_16k))]


def test_the_budget_bounds_how_long_silence_is_fed():
    engine = _engine(budget_ms=100)  # five 20 ms frames

    for _ in range(10):
        engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)

    assert len(_drain(engine)) == 5


def test_a_zero_budget_restores_dropping():
    engine = _engine(budget_ms=0)

    engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)

    assert _drain(engine) == []


def test_the_default_budget_applies_when_the_call_is_unknown():
    engine = _engine()
    frames_in_budget = int(PIPELINE_GATED_SILENCE_MS_DEFAULT / 20)

    for _ in range(frames_in_budget + 10):
        engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)

    assert len(_drain(engine)) == frames_in_budget


def test_a_missing_queue_is_not_an_error():
    engine = _engine()
    engine._pipeline_queues.clear()

    engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)


def test_an_empty_frame_is_ignored():
    engine = _engine()

    engine._feed_pipeline_silence(CALL_ID, b"", 8000)

    assert _drain(engine) == []


def test_a_full_queue_is_not_an_error():
    engine = _engine(maxsize=2)
    q = engine._pipeline_queues[CALL_ID]
    while not q.full():
        q.put_nowait(b"x")

    engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)


def test_the_budget_is_spent_per_gated_stretch_not_per_call():
    """The consumer loop clears the tally as soon as real audio flows again."""
    engine = _engine(budget_ms=40)

    for _ in range(4):
        engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)
    assert len(_drain(engine)) == 2

    engine._pipeline_gated_silence_used_ms.pop(CALL_ID, None)
    for _ in range(4):
        engine._feed_pipeline_silence(CALL_ID, FRAME_8K, 8000)

    assert len(_drain(engine)) == 2
