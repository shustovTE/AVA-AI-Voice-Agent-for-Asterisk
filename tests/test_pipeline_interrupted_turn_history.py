"""An interrupted streaming turn keeps the caller's words in the history.

The streaming overlap path appended the exchange only after the whole reply
had been spoken. A barge-in raised _PipelinePlaybackInterrupted before that
point and the turn returned with nothing recorded, so the model answered the
caller's next words with no memory of what they had just said — in a
production call it re-asked a question the caller had already answered.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent, TTSComponent
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _StubResolution

CALL_ID = "call-interrupted"
CALLER_SAID = "я хочу полный капитальный ремонт у меня три комнаты"
SENTENCES = ["Первое предложение.", "Второе предложение.", "Третье предложение."]


def _config() -> AppConfig:
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
            "downstream_mode": "stream",
            "streaming": {"pipeline_streaming_overlap": True},
        }
    )


class _StreamingLLM(LLMComponent):
    supports_streaming = True

    async def generate(self, call_id, transcript, context, options):
        return " ".join(SENTENCES)

    async def generate_stream(self, call_id, transcript, context, options):
        for sentence in SENTENCES:
            for word in sentence.split(" "):
                yield word + " "


class _PlaybackStub:
    """Owns the stream; flipping ``active`` is what a barge-in does."""

    def __init__(self):
        self.active = True
        self.stopped = asyncio.Event()

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.queue = queue
        return "stream-1"

    def is_stream_active(self, call_id, stream_id=None):
        return self.active and stream_id == "stream-1"

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.active = False
        self.stopped.set()


class _InterruptingTTS(TTSComponent):
    """Cuts the stream while synthesizing the sentence at ``interrupt_at``."""

    downstream_mode_override = "stream"

    def __init__(self, playback, interrupt_at):
        self.playback = playback
        self.interrupt_at = interrupt_at
        self.synthesized = []

    async def synthesize(self, call_id, text, options):
        self.synthesized.append(text)
        if len(self.synthesized) == self.interrupt_at:
            self.playback.active = False
        yield b"\x00" * 320


async def _run_turn(monkeypatch, interrupt_at):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    tts = _InterruptingTTS(playback, interrupt_at)
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=_StreamingLLM(),
        tts_adapter=tts,
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(
        engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution
    )
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=CALL_ID, caller_channel_id=CALL_ID)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)

    await stt.results.put(CALLER_SAID)
    await asyncio.wait_for(playback.stopped.wait(), timeout=3)
    await asyncio.sleep(0.1)
    updated = await engine.session_store.get_by_call_id(CALL_ID)
    history = [(m["role"], m["content"]) for m in (updated.conversation_history or [])]
    await engine._cleanup_call(CALL_ID)
    return history, tts


@pytest.mark.asyncio
async def test_interruption_mid_reply_keeps_the_caller_and_the_spoken_part(monkeypatch):
    history, tts = await _run_turn(monkeypatch, interrupt_at=2)

    assert ("user", CALLER_SAID) in history
    assert ("assistant", SENTENCES[0]) in history
    assert not any(SENTENCES[1] in content for _, content in history)
    assert not any(SENTENCES[2] in content for _, content in history)


@pytest.mark.asyncio
async def test_interruption_before_any_sentence_keeps_only_the_caller(monkeypatch):
    history, _ = await _run_turn(monkeypatch, interrupt_at=1)

    assert history == [("user", CALLER_SAID)]


@pytest.mark.asyncio
async def test_the_stream_was_actually_cut_short(monkeypatch):
    """Guards the harness: without an interruption all sentences are synthesized."""
    _, tts = await _run_turn(monkeypatch, interrupt_at=2)

    assert tts.synthesized == SENTENCES[:2]
