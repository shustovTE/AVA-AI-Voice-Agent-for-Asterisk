"""Each pipeline turn records its stage latencies on its history entry.

The call record kept one number per turn, the final transcript to the reply's
first audio, and only as the call's average and maximum. A slow turn could not
be traced to its stage: the recognizer, the model's first token or first
sentence, or the speech synthesis. Each assistant entry of the conversation
now carries a ``latency`` object with the stages the turn measured, which the
Admin UI shows under the reply; the model never sees it.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.core.utterances import SttUtterance
from src.engine import Engine, _sanitize_for_llm
from src.pipelines.base import LLMComponent, STTComponent, TTSComponent
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _StubResolution

CALL_ID = "call-latency"
CALLER_SAID = "мне нужен ремонт кухни"
REPLY = "Первое предложение. Второе предложение."
FIRST_TOKEN_DELAY = 0.03
FIRST_AUDIO_DELAY = 0.02
ASR_WAIT = 0.05


def _config(*, overlap: bool, greeting: str = "") -> AppConfig:
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
            "llm": {"initial_greeting": greeting, "prompt": "You are helpful", "model": "gpt-4o"},
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {
                "pipeline_streaming_overlap": overlap,
                "pipeline_heard_reply_lead_ms": 0,
            },
        }
    )


class _TimedLLM(LLMComponent):
    """Answers after a pause before its first token; records what it was shown."""

    supports_streaming = True

    def __init__(self):
        self.contexts = []

    async def generate(self, call_id, transcript, context, options):
        self.contexts.append(context)
        await asyncio.sleep(FIRST_TOKEN_DELAY)
        return REPLY

    async def generate_stream(self, call_id, transcript, context, options):
        self.contexts.append(context)
        await asyncio.sleep(FIRST_TOKEN_DELAY)
        for word in REPLY.split(" "):
            yield word + " "


class _TimedTTS(TTSComponent):
    """One frame per request, after a pause."""

    downstream_mode_override = "stream"

    def __init__(self):
        self.synthesized = []

    async def synthesize(self, call_id, text, options):
        self.synthesized.append(text)
        await asyncio.sleep(FIRST_AUDIO_DELAY)
        yield b"\x00" * 320


class _SlowChunkedSTT(STTComponent):
    """A buffered recognizer that takes its time over each chunk."""

    async def transcribe(self, call_id, audio_pcm16, sample_rate_hz, options):
        await asyncio.sleep(ASR_WAIT)
        return CALLER_SAID


class _DrainingPlayback:
    """Owns the stream and drains it; the end sentinel ends the stream."""

    def __init__(self):
        self.active = False
        self.finished = asyncio.Event()
        self.queue = None
        self._drainer = None

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.queue = queue
        self.active = True
        self.finished.clear()
        self._drainer = asyncio.create_task(self._drain(queue))
        return "stream-1"

    async def _drain(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                self.active = False
                self.finished.set()
                return

    def is_stream_active(self, call_id, stream_id=None):
        return self.active and stream_id in (None, "stream-1")

    def get_playback_position_ms(self, call_id):
        return 0

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.active = False
        self.finished.set()


async def _start(monkeypatch, *, overlap, stt=None, stt_options=None, greeting=""):
    engine = Engine(_config(overlap=overlap, greeting=greeting))
    engine.pipeline_orchestrator._started = True
    playback = _DrainingPlayback()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = stt or _ResultStreamingStubSTT()
    llm = _TimedLLM()
    tts = _TimedTTS()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options=stt_options or {"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=tts,
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=CALL_ID, caller_channel_id=CALL_ID)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    return engine, session, stt, llm, playback, resolution


def _utterance() -> SttUtterance:
    now = time.monotonic()
    return SttUtterance(
        pcm16=b"\x01\x00" * 8000,
        sample_rate=16000,
        utterance_id="utt-1",
        started_at=now - 0.5,
        ended_at=now,
        signal_ms=500.0,
    )


async def _answered_turn(engine, session, stt, playback):
    """Send one caller utterance, its result, and wait for the reply to be played."""
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    assert engine._send_pipeline_utterance(CALL_ID, _utterance(), expect_result=True)
    await asyncio.sleep(ASR_WAIT)
    await stt.results.put(CALLER_SAID)
    await asyncio.wait_for(playback.finished.wait(), timeout=3)
    await asyncio.sleep(0.05)
    updated = await engine.session_store.get_by_call_id(CALL_ID)
    return list(updated.conversation_history or [])


def _reply_entry(history):
    [entry] = [m for m in history if m.get("role") == "assistant"]
    return entry


@pytest.mark.asyncio
async def test_streaming_overlap_turn_records_every_stage(monkeypatch):
    engine, session, stt, llm, playback, _ = await _start(monkeypatch, overlap=True)
    try:
        history = await _answered_turn(engine, session, stt, playback)
    finally:
        await engine._cleanup_call(CALL_ID)

    latency = _reply_entry(history)["latency"]
    assert set(latency) == {"asr_ms", "llm_first_token_ms", "llm_ms", "tts_ms", "turn_ms"}
    assert all(isinstance(v, int) for v in latency.values())
    # The recognizer's time: from the utterance's hand-off to its result.
    assert latency["asr_ms"] >= ASR_WAIT * 1000 * 0.8
    # The model's: to its first token, and to the first sentence the TTS started on.
    assert latency["llm_first_token_ms"] >= FIRST_TOKEN_DELAY * 1000 * 0.8
    assert latency["llm_ms"] >= latency["llm_first_token_ms"]
    # The synthesis's: to its first audio; the turn spans both.
    assert latency["tts_ms"] >= FIRST_AUDIO_DELAY * 1000 * 0.8
    assert latency["turn_ms"] + 2 >= latency["llm_ms"] + latency["tts_ms"]
    # The caller's entry carries none, and the model is never shown it.
    [caller] = [m for m in history if m.get("role") == "user"]
    assert "latency" not in caller
    assert all("latency" not in m for m in _sanitize_for_llm(history))
    assert CALL_ID not in engine._pipeline_pending_asr_ms


@pytest.mark.asyncio
async def test_serial_turn_records_the_whole_generation_and_its_tts(monkeypatch):
    engine, session, stt, llm, playback, _ = await _start(monkeypatch, overlap=False)
    try:
        history = await _answered_turn(engine, session, stt, playback)
    finally:
        await engine._cleanup_call(CALL_ID)

    entry = _reply_entry(history)
    assert entry["content"] == REPLY
    latency = entry["latency"]
    # No token stream in the serial path: no first-token figure.
    assert set(latency) == {"asr_ms", "llm_ms", "tts_ms", "turn_ms"}
    assert latency["llm_ms"] >= FIRST_TOKEN_DELAY * 1000 * 0.8
    # The entry is recorded before the TTS starts; the TTS stages still land on it.
    assert latency["tts_ms"] >= FIRST_AUDIO_DELAY * 1000 * 0.8
    assert latency["turn_ms"] + 2 >= latency["llm_ms"] + latency["tts_ms"]


@pytest.mark.asyncio
async def test_chunked_recognizer_turn_measures_its_transcribe_call(monkeypatch):
    engine, session, stt, llm, playback, _ = await _start(
        monkeypatch, overlap=False, stt=_SlowChunkedSTT(), stt_options={"streaming": False, "chunk_ms": 80}
    )
    try:
        # One commit of 16 kHz PCM16 audio (80 ms) is transcribed as a whole.
        await engine._pipeline_queues[CALL_ID].put(b"\x01\x00" * 1280)
        await asyncio.wait_for(playback.finished.wait(), timeout=3)
        await asyncio.sleep(0.05)
        updated = await engine.session_store.get_by_call_id(CALL_ID)
        history = list(updated.conversation_history or [])
    finally:
        await engine._cleanup_call(CALL_ID)

    latency = _reply_entry(history)["latency"]
    assert latency["asr_ms"] >= ASR_WAIT * 1000 * 0.8
    assert {"llm_ms", "tts_ms", "turn_ms"} <= set(latency)


@pytest.mark.asyncio
async def test_greeting_records_its_tts_share_only(monkeypatch):
    engine, session, stt, llm, playback, _ = await _start(monkeypatch, overlap=True, greeting="Алло.")
    try:
        await asyncio.wait_for(stt.started.wait(), timeout=2)
        await asyncio.wait_for(playback.finished.wait(), timeout=3)
        [greeting] = [m for m in session.conversation_history if m.get("role") == "assistant"]
    finally:
        await engine._cleanup_call(CALL_ID)

    assert greeting["content"] == "Алло."
    assert set(greeting["latency"]) == {"tts_ms"}
    assert greeting["latency"]["tts_ms"] >= FIRST_AUDIO_DELAY * 1000 * 0.8


@pytest.mark.asyncio
async def test_tool_continuation_tts_reports_its_first_audio(monkeypatch):
    engine, session, stt, llm, playback, resolution = await _start(monkeypatch, overlap=True)
    try:
        timing = {}
        await engine._stream_pipeline_tts_text(CALL_ID, session, resolution, "Готово.", timing=timing)
    finally:
        await engine._cleanup_call(CALL_ID)

    assert set(timing) == {"tts_ms"}
    assert timing["tts_ms"] >= FIRST_AUDIO_DELAY * 1000 * 0.8


@pytest.mark.asyncio
async def test_final_arrival_keeps_the_recognizer_latency_for_the_turn():
    engine = Engine(_config(overlap=True))

    # Measured from the utterance's hand-off when nothing better is known.
    engine._pipeline_stt_final_expected_at["c1"] = time.monotonic() - 0.1
    engine._note_pipeline_final_arrived("c1")
    assert 80 <= engine._pipeline_pending_asr_ms["c1"] <= 2000
    assert "c1" not in engine._pipeline_stt_final_expected_at

    # A caller that measured the recognizer itself overrides the hand-off clock.
    engine._pipeline_stt_final_expected_at["c2"] = time.monotonic() - 5.0
    engine._note_pipeline_final_arrived("c2", asr_ms=42.4)
    assert engine._pipeline_pending_asr_ms["c2"] == 42

    # A result nobody timed (the recognizer segments the stream itself) records nothing.
    engine._note_pipeline_final_arrived("c3")
    assert "c3" not in engine._pipeline_pending_asr_ms
    assert "c3" in engine._pipeline_last_final_at
