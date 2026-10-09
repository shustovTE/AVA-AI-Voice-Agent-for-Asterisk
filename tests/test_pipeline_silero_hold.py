"""With Silero as the turn source, a pending turn is held while Silero hears speech.

The hold on a pending result while the caller talks ran from the recognizer's
last result: a safety valve for Asterisk talk detection, whose end event can be
lost, on the assumption that a talking caller produces a result every phrase.
A VAD-gated recognizer (GigaAM v3, Sherpa offline) gives no result for as long
as a sentence lasts, so a caller in a sentence longer than the hold had the
turn released under them: the model answered the previous phrase while they
were still talking. Silero is the engine's own detector and its stop cannot
be lost, so its hold now runs from the last frame it scored as speech.
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
            "pipelines": {"serial": {}},
            "active_pipeline": "serial",
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {"pipeline_streaming_overlap": False},
        }
    )


class _CountingLLM(LLMComponent):
    supports_streaming = False

    def __init__(self):
        self.calls = []

    async def generate(self, call_id, transcript, context, options):
        self.calls.append(transcript)
        return "ответ"


class _SilentTTS(TTSComponent):
    downstream_mode_override = "stream"

    async def synthesize(self, call_id, text, options):
        yield b"\x00" * 320


class _PlaybackStub:
    async def start_streaming_playback(self, call_id, queue, **kwargs):
        return "stream-1"

    def is_stream_active(self, call_id, stream_id=None):
        return False

    def get_playback_position_ms(self, call_id):
        return 0

    async def stop_streaming_playback(self, call_id, *, drain=False):
        return True


async def _start(monkeypatch, *, silero: bool):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    monkeypatch.setattr(engine, "streaming_playback_manager", _PlaybackStub())
    stt = _ResultStreamingStubSTT()
    llm = _CountingLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=_SilentTTS(),
    )
    resolution.llm_options = {
        "end_of_turn_talk_detect_hold_ms": 300,
        "end_of_turn_talk_detect_grace_ms": 100,
    }
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    call_id = f"call-hold-{uuid4().hex[:8]}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "serial"
    await engine.session_store.upsert_call(session)
    tracker = None
    if silero:
        # Stands in for the SileroCallerTracker the runner would build with Silero enabled.
        tracker = SimpleNamespace(last_speech_at=time.monotonic(), talking=True)
        engine._silero_trackers[call_id] = tracker
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, session, stt, llm, tracker


@pytest.mark.asyncio
async def test_a_long_sentence_holds_the_turn_while_silero_hears_speech(monkeypatch):
    engine, session, stt, llm, tracker = await _start(monkeypatch, silero=True)
    engine._note_pipeline_caller_talking(session.call_id, True, source="vad")
    await stt.results.put("подождите")  # the first phrase; the caller goes on for longer than the hold

    async def keep_talking():
        for _ in range(20):
            tracker.last_speech_at = time.monotonic()
            await asyncio.sleep(0.05)

    await keep_talking()  # one second of speech frames, three times the 300 ms hold
    assert llm.calls == []

    engine._note_pipeline_caller_talking(session.call_id, False, source="vad")
    for _ in range(40):
        if llm.calls:
            break
        await asyncio.sleep(0.05)
    assert llm.calls == ["подождите"]
    await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_detector_left_talking_without_speech_frames_still_releases(monkeypatch):
    """Audio that stopped flowing leaves Silero "talking" with no new frames: the old valve applies."""
    engine, session, stt, llm, tracker = await _start(monkeypatch, silero=True)
    engine._note_pipeline_caller_talking(session.call_id, True, source="vad")
    tracker.last_speech_at = time.monotonic()
    await stt.results.put("алло")

    for _ in range(40):
        if llm.calls:
            break
        await asyncio.sleep(0.05)
    assert llm.calls == ["алло"]
    await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_talk_detection_keeps_its_hold_from_the_last_result(monkeypatch):
    engine, session, stt, llm, _ = await _start(monkeypatch, silero=False)
    engine._note_pipeline_caller_talking(session.call_id, True, source="talk_detect")
    started = time.monotonic()
    await stt.results.put("алло")

    for _ in range(40):
        if llm.calls:
            break
        await asyncio.sleep(0.05)
    assert llm.calls == ["алло"]
    assert 0.25 <= time.monotonic() - started < 1.5
    await engine._cleanup_call(session.call_id)
