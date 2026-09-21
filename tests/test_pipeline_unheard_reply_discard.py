"""A reply the caller talks over before its first sound is discarded, not played.

The end of a caller's turn is a guess on a pause. When the guess was wrong,
the caller goes on within a second, while the model is still answering the
half-sentence. Nothing of that answer has reached them, so it is dropped: the
LLM request is cancelled, no TTS is requested, a stream that has not played
is stopped, and the caller's words are held to be answered together with what
they say next, as one turn. Words spoken into a reply that is already audible
wait for it to end instead of cutting it (barge-in does that); words from
before the reply started still cut it, as before.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent, TTSComponent
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _StubResolution


def _config(streaming=None) -> AppConfig:
    return AppConfig(
        **{
            "default_provider": "local",
            "providers": {"local": {"enabled": True}},
            "asterisk": {"host": "127.0.0.1", "port": 8088, "username": "u", "password": "p", "app_name": "a"},
            "llm": {"initial_greeting": "", "prompt": "You are helpful", "model": "gpt-4o"},
            "pipelines": {"serial": {}},
            "active_pipeline": "serial",
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {
                "pipeline_streaming_overlap": False,
                "pipeline_heard_reply_on_interrupt": True,
                "pipeline_heard_reply_lead_ms": 0,
                **(streaming or {}),
            },
        }
    )


class _GatedLLM(LLMComponent):
    """Answers only when released; records cancellations."""

    supports_streaming = False

    def __init__(self, *, block=True):
        self.block = block
        self.transcripts = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = 0
        self.cancel_requests = []

    async def generate(self, call_id, transcript, context, options):
        self.transcripts.append(transcript)
        self.started.set()
        if self.block:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return f"ответ на: {transcript}"

    async def cancel_generation(self, call_id):
        self.cancel_requests.append(call_id)


class _SlowTTS(TTSComponent):
    downstream_mode_override = "stream"

    def __init__(self, chunks=1, delay=0.0):
        self.chunks = chunks
        self.delay = delay
        self.requests = 0

    async def synthesize(self, call_id, text, options):
        self.requests += 1
        for _ in range(self.chunks):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield b"\x00" * 320


class _PlaybackStub:
    def __init__(self):
        self.active = False
        self.current = None
        self.starts = []
        self.stops = 0
        self.position_ms = 0
        self.active_streams = {}

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.current = f"stream-{len(self.starts) + 1}"
        self.starts.append(self.current)
        self.active = True
        self.active_streams[call_id] = {"stream_id": self.current, "playback_type": kwargs.get("playback_type")}
        return self.current

    def is_stream_active(self, call_id, stream_id=None):
        return self.active and (stream_id is None or stream_id == self.current)

    def get_playback_position_ms(self, call_id):
        return self.position_ms

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.stops += 1
        self.active = False
        self.active_streams.pop(call_id, None)
        return True


async def _start(monkeypatch, *, llm, tts=None, streaming=None):
    engine = Engine(_config(streaming))
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=tts or _SlowTTS(),
    )
    resolution.llm_options = {"end_of_turn_talk_detect_grace_ms": 50, "end_of_turn_talk_detect_hold_ms": 400}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    call_id = f"call-unheard-{uuid4().hex[:8]}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "serial"
    session.media_rx_confirmed = True
    await engine.session_store.upsert_call(session)
    # Stands in for the Silero tracker the runner builds with Silero enabled.
    tracker = SimpleNamespace(
        last_speech_at=None, talking=False, segment_started_at=None, last_probability=0.0
    )
    engine._silero_trackers[call_id] = tracker
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, session, stt, playback, tracker


async def _caller_resumes(engine, session, tracker):
    tracker.talking = True
    tracker.last_speech_at = time.monotonic()
    tracker.segment_started_at = time.monotonic()
    await engine._on_silero_speech_started(session, tracker, source="test")


def _caller_stops(engine, session, tracker):
    tracker.talking = False
    engine._note_pipeline_caller_talking(session.call_id, False, source="vad")


async def _wait_for(predicate, timeout=3.0):
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
async def test_the_caller_going_on_during_generation_cancels_it_and_merges_the_turn(monkeypatch):
    llm = _GatedLLM()
    tts = _SlowTTS()
    engine, session, stt, playback, tracker = await _start(monkeypatch, llm=llm, tts=tts)
    try:
        await stt.results.put("у меня три")
        await asyncio.wait_for(llm.started.wait(), timeout=2)

        await _caller_resumes(engine, session, tracker)  # ... "комнаты и кухня"
        assert await _wait_for(lambda: llm.cancelled == 1)
        assert llm.cancel_requests == [session.call_id]
        assert tts.requests == 0 and playback.starts == []

        llm.block = False
        _caller_stops(engine, session, tracker)
        await stt.results.put("комнаты и кухня")
        assert await _wait_for(lambda: len(llm.transcripts) == 2)
        assert llm.transcripts == ["у меня три", "у меня три комнаты и кухня"]
        assert await _wait_for(lambda: len(session.conversation_history) == 2)
        assert [m["role"] for m in session.conversation_history] == ["user", "assistant"]
        assert session.conversation_history[0]["content"] == "у меня три комнаты и кухня"
        assert playback.starts == ["stream-1"]
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_reply_whose_stream_has_not_played_is_dropped_with_its_history(monkeypatch):
    llm = _GatedLLM(block=False)
    tts = _SlowTTS(chunks=3, delay=0.15)
    engine, session, stt, playback, tracker = await _start(monkeypatch, llm=llm, tts=tts)
    try:
        await stt.results.put("подождите")
        assert await _wait_for(lambda: playback.starts == ["stream-1"])
        assert playback.position_ms == 0  # nothing has reached the caller

        await _caller_resumes(engine, session, tracker)
        assert await _wait_for(lambda: playback.stops == 1)
        assert engine._spoken_replies.get(session.call_id) is None
        assert await _wait_for(lambda: session.call_id not in engine._pipeline_reply_inflight)
        assert session.conversation_history == []

        _caller_stops(engine, session, tracker)
        await stt.results.put("я ещё думаю")
        assert await _wait_for(lambda: len(llm.transcripts) == 2)
        assert llm.transcripts[-1] == "подождите я ещё думаю"
        assert await _wait_for(lambda: len(session.conversation_history) == 2)
        assert session.conversation_history[0]["content"] == "подождите я ещё думаю"
        assert playback.starts == ["stream-1", "stream-2"]
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_reply_already_audible_is_left_to_barge_in(monkeypatch):
    llm = _GatedLLM(block=False)
    tts = _SlowTTS(chunks=3, delay=0.15)
    engine, session, stt, playback, tracker = await _start(monkeypatch, llm=llm, tts=tts)
    try:
        await stt.results.put("подождите")
        assert await _wait_for(lambda: playback.starts == ["stream-1"])
        playback.position_ms = 120  # the first sound is out
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time()

        await _caller_resumes(engine, session, tracker)
        await asyncio.sleep(0.3)
        assert playback.stops == 0
        assert not engine._pipeline_reply_superseded(session.call_id)
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_the_discard_can_be_switched_off(monkeypatch):
    llm = _GatedLLM()
    engine, session, stt, playback, tracker = await _start(
        monkeypatch, llm=llm, streaming={"pipeline_discard_unheard_reply": False}
    )
    try:
        await stt.results.put("у меня три")
        await asyncio.wait_for(llm.started.wait(), timeout=2)
        await _caller_resumes(engine, session, tracker)
        await asyncio.sleep(0.2)
        assert llm.cancelled == 0
        llm.release.set()
        assert await _wait_for(lambda: playback.starts == ["stream-1"])
        assert llm.transcripts == ["у меня три"]
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_words_spoken_into_an_audible_reply_wait_for_it_to_end(monkeypatch):
    llm = _GatedLLM(block=False)
    engine, session, stt, playback, tracker = await _start(monkeypatch, llm=llm)
    try:
        # A reply is audible; the caller answered it inside the barge-in protection.
        playback.active = True
        playback.current = "stream-greeting"
        playback.position_ms = 800
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time() - 1.0
        tracker.segment_started_at = time.monotonic()  # after the reply started
        await stt.results.put("да, слушаю")
        await asyncio.sleep(0.5)
        assert llm.transcripts == []  # held, not answered on top of the reply
        assert playback.stops == 0  # and not cut

        # The reply ends.
        playback.active = False
        session.audio_capture_enabled = True
        session.tts_playing = False
        assert await _wait_for(lambda: llm.transcripts == ["да, слушаю"])
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_words_from_before_the_reply_started_still_cut_it(monkeypatch):
    llm = _GatedLLM(block=False)
    engine, session, stt, playback, tracker = await _start(monkeypatch, llm=llm)
    try:
        playback.active = True
        playback.current = "stream-late"
        playback.active_streams[session.call_id] = {"stream_id": "stream-late", "playback_type": "pipeline-tts"}
        playback.position_ms = 800
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time()
        tracker.segment_started_at = time.monotonic() - 3.0  # the recognizer was late
        await stt.results.put("это я говорил раньше")
        assert await _wait_for(lambda: llm.transcripts == ["это я говорил раньше"])
        assert playback.stops >= 1
    finally:
        await engine._cleanup_call(session.call_id)
