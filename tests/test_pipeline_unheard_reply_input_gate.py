"""Discard Off commits to one reply and drops input until it becomes audible."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.core.models import CallSession
from src.audio.audiosocket_protocol import AudioSocketAudioFrame
from src.core.utterances import SttUtterance
from src.engine import Engine
from tests.test_pipeline_unheard_reply_discard import (
    _config, _start, _GatedLLM, _SlowTTS, _caller_resumes, _wait_for,
)
from tests.test_pipeline_utterance_stt import _start_call, _hear, _UtteranceStubSTT


OFF = {"pipeline_discard_unheard_reply": False}


class _GatedStreamingLLM(_GatedLLM):
    supports_streaming = True

    async def generate_stream(self, call_id, transcript, context, options):
        yield (await self.generate(call_id, transcript, context, options)) + "."


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", [False, True])
async def test_off_drops_recognition_during_generation_instead_of_queueing_a_second_reply(monkeypatch, overlap):
    llm = _GatedStreamingLLM() if overlap else _GatedLLM()
    engine, session, stt, playback, tracker = await _start(
        monkeypatch, llm=llm, streaming={**OFF, "pipeline_streaming_overlap": overlap}
    )
    try:
        await stt.results.put("первый вопрос")
        await asyncio.wait_for(llm.started.wait(), 2)
        await _caller_resumes(engine, session, tracker)
        await stt.results.put("слова во время генерации")
        assert await _wait_for(stt.results.empty)
        assert llm.cancelled == 0
        assert not engine._pipeline_caller_resumed[session.call_id].is_set()

        tracker.talking = False
        llm.release.set()
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        playback.active = False
        await asyncio.sleep(0.2)
        assert llm.transcripts == ["первый вопрос"]
        assert all("слова во время" not in str(m) for m in session.conversation_history)

        await stt.results.put("новый вопрос после ответа")
        assert await _wait_for(lambda: len(llm.transcripts) == 2)
        assert llm.transcripts[-1] == "новый вопрос после ответа"
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_off_stays_closed_through_tts_and_buffered_playback(monkeypatch):
    engine, session, stt, playback, tracker = await _start(
        monkeypatch, llm=_GatedLLM(block=False), tts=_SlowTTS(chunks=3, delay=0.1), streaming=OFF
    )
    try:
        await stt.results.put("вопрос")
        assert await _wait_for(lambda: bool(playback.starts))
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time() - 10  # synthesis spent the old wall-clock window
        await _caller_resumes(engine, session, tracker)
        assert playback.stops == 0
        assert engine._pipeline_input_blocked_before_reply(session)
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        assert engine._pipeline_input_blocked_before_reply(session)

        playback.position_ms = 100
        assert not engine._pipeline_input_blocked_before_reply(session)
        engine._apply_barge_in_action = AsyncMock()
        assert await engine._silero_barge_in(session, tracker, source="test") == "protected"
        engine._apply_barge_in_action.assert_not_awaited()

        playback.position_ms = 1700
        assert await engine._silero_barge_in(session, tracker, source="test") == "fired"
        engine._apply_barge_in_action.assert_awaited_once()
        assert session.audio_capture_enabled
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_off_drops_late_whole_utterance_result_even_after_playback(monkeypatch):
    llm = _GatedLLM()
    stt = _UtteranceStubSTT()
    engine, session, stt, playback, tracker = await _start(monkeypatch, llm=llm, streaming=OFF, stt=stt)
    try:
        queue = engine._pipeline_queues[session.call_id]
        def utterance(n):
            return SttUtterance(b"\x11\x22" * 1600, 16000, str(n), 0, 0.1, signal_ms=100)
        await queue.put(utterance(1))
        await queue.put(utterance(2))
        assert await _wait_for(lambda: len(stt.utterances) == 2)
        await stt.results.put("вопрос")
        await asyncio.wait_for(llm.started.wait(), 2)
        llm.release.set()
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        playback.active = False
        await stt.results.put("запоздавшая расшифровка")
        assert await _wait_for(stt.results.empty)
        await asyncio.sleep(0.2)
        assert llm.transcripts == ["вопрос"]

        await queue.put(utterance(3))
        assert await _wait_for(lambda: len(stt.utterances) == 3)
        await stt.results.put("следующий вопрос")
        assert await _wait_for(lambda: len(llm.transcripts) == 2)
        assert llm.transcripts[-1] == "следующий вопрос"
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("listen_during_playback", [False, True])
async def test_blocked_audio_never_enters_silero_or_preroll(monkeypatch, listen_during_playback):
    stt = _UtteranceStubSTT()
    engine, session, model, _ = await _start_call(
        monkeypatch, stt=stt, streaming=OFF,
        barge_in={"pipeline_listen_during_playback": listen_during_playback},
    )
    try:
        cutter = engine._utterance_cutters[session.call_id]
        cutter.append(b"\x33\x44" * 512)
        cutter.speech_started()
        engine._begin_pipeline_reply(session.call_id)
        assert not cutter.open
        before = cutter.position
        await _hear(engine, session, model, [0.9] * 4 + [0.1] * 4)
        assert cutter.position == before
        assert not engine._silero_trackers[session.call_id].talking
        assert stt.utterances == []
        assert not engine._pipeline_caller_resumed[session.call_id].is_set()

        engine._end_pipeline_reply(session.call_id)
        model.probabilities.clear()
        await _hear(engine, session, model, [0.9] * 4 + [0.1] * 4)
        assert await _wait_for(lambda: len(stt.utterances) == 1)
        assert b"\x33\x44" not in stt.utterances[0]["audio"]
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "empty", "cancel"])
async def test_input_reopens_when_generation_ends_without_audio(monkeypatch, failure):
    llm = _GatedLLM()
    engine, session, stt, playback, _ = await _start(monkeypatch, llm=llm, streaming=OFF)
    async def fail(*args):
        llm.started.set()
        await llm.release.wait()
        if failure == "empty":
            return ""
        if failure == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("upstream unavailable")
    monkeypatch.setattr(llm, "generate", fail)
    try:
        await stt.results.put("вопрос")
        await asyncio.wait_for(llm.started.wait(), 2)
        assert engine._pipeline_input_blocked_before_reply(session)
        llm.release.set()
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        assert not engine._pipeline_input_blocked_before_reply(session)
        assert not playback.starts
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["externalmedia", "audiosocket"])
async def test_transport_drops_speech_before_the_first_sound(transport):
    engine = Engine(_config(OFF))
    session = CallSession(call_id="input-gate", caller_channel_id="input-gate")
    session.media_rx_confirmed = True
    await engine.session_store.upsert_call(session)
    engine._pipeline_forced[session.call_id] = True
    engine._pipeline_queues[session.call_id] = asyncio.Queue()
    engine._begin_pipeline_reply(session.call_id)
    engine._observe_silero_vad = AsyncMock()
    engine._silero_vad_active = lambda _: True
    if transport == "externalmedia":
        await engine._on_rtp_audio(session.call_id, 1, b"\x11\x22" * 320)
    else:
        engine.conn_to_channel["conn"] = session.call_id
        await engine._audiosocket_handle_audio(
            "conn", AudioSocketAudioFrame(b"\x11\x22" * 160, 0x10, "slin", 8000)
        )
    engine._observe_silero_vad.assert_not_awaited()
    queue = engine._pipeline_queues[session.call_id]
    assert not queue.empty()  # the STT timeline receives silence, never the caller's words
    while not queue.empty():
        assert not any(queue.get_nowait())
