"""What the caller heard of an interrupted pipeline reply (src/core/heard_reply.py)."""

import pytest
from pydantic import ValidationError

from src.config import StreamingConfig
from src.core.heard_reply import ELLIPSIS, SpokenReply, word_prefix


def test_word_prefix_cuts_back_to_a_word_boundary():
    text = "Второе предложение, довольно длинное."
    assert word_prefix(text, 0.0) == ""
    assert word_prefix(text, 1.0) == text
    assert word_prefix("Второе предложение.", 0.5) == "Второе"
    assert word_prefix(text, 0.55) == "Второе предложение"  # the trailing comma goes too
    assert word_prefix("Слово", 0.5) == ""  # less than one word: nothing


def _reply(lead_ms: float = 0.0) -> SpokenReply:
    return SpokenReply(call_id="c", stream_id="s", bytes_per_ms=8.0, lead_ms=lead_ms)


def test_whole_sentences_plus_a_proportional_prefix_of_the_cut_one():
    reply = _reply()
    for sentence in ("Первое предложение.", "Второе предложение.", "Третье предложение."):
        reply.open_segment(sentence)
        reply.add_audio_bytes(320)  # 40 ms of mu-law at 8 kHz
        reply.close_segment()
    reply.completed = True

    assert reply.heard_text_at(60) == "Первое предложение. Второе" + ELLIPSIS
    assert reply.heard_text_at(40) == "Первое предложение. " + ELLIPSIS  # cut on the sentence boundary
    assert reply.heard_text_at(0) == ""
    assert reply.heard_text_at(120) == "Первое предложение. Второе предложение. Третье предложение."
    assert reply.heard_text_at(500) == reply.full_text


def test_a_reply_still_being_generated_is_marked_even_when_all_queued_audio_played():
    reply = _reply()
    reply.open_segment("Первое предложение.")
    reply.add_audio_bytes(320)
    reply.close_segment()
    assert reply.heard_text_at(40) == "Первое предложение. " + ELLIPSIS


def test_the_lead_is_subtracted_from_the_played_position():
    reply = _reply(lead_ms=200.0)
    reply.open_segment("Первое предложение.")
    reply.add_audio_bytes(320 * 25)  # 1000 ms
    reply.close_segment()
    reply.completed = True
    # 800 of 1000 ms heard lands inside the second word; the boundary before it is
    # the single space after "Первое", so only the first word is kept.
    assert reply.heard_text_at(1000) == "Первое" + ELLIPSIS
    assert reply.heard_text_at(1200) == "Первое предложение."


def test_an_open_segment_is_measured_by_the_speech_rate_of_the_completed_ones():
    reply = _reply()
    reply.open_segment("Первое предложение.")  # 19 chars
    reply.add_audio_bytes(320)  # 40 ms -> ~2.1 ms per char
    reply.close_segment()
    reply.open_segment("Второе предложение.")  # still synthesizing: no audio yet
    assert reply.heard_text_at(60) == "Первое предложение. Второе" + ELLIPSIS


def test_an_open_segment_without_history_uses_the_default_rate():
    reply = _reply()
    reply.open_segment("abcdefghij klmnop")  # 17 chars * 65 ms = 1105 ms expected
    assert reply.heard_text_at(1105 * 0.9) == "abcdefghij" + ELLIPSIS
    assert reply.heard_text_at(200) == ""


def test_mark_interrupted_records_the_position_and_the_text():
    reply = _reply()
    reply.open_segment("Первое предложение.")
    reply.add_audio_bytes(320)
    reply.close_segment()
    reply.completed = True
    assert reply.mark_interrupted(20) == "Первое" + ELLIPSIS
    assert reply.interrupted is True and reply.played_ms == 20 and reply.heard_text == "Первое" + ELLIPSIS


def test_config_defaults_and_bounds():
    cfg = StreamingConfig()
    assert cfg.pipeline_heard_reply_on_interrupt is True
    assert cfg.pipeline_heard_reply_lead_ms == 200
    assert StreamingConfig(pipeline_heard_reply_on_interrupt=False, pipeline_heard_reply_lead_ms=0).pipeline_heard_reply_lead_ms == 0
    with pytest.raises(ValidationError):
        StreamingConfig(pipeline_heard_reply_lead_ms=6000)
