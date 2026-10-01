"""A pipeline greeting the caller cuts does not hold the call hostage.

Reproduces call 1790834848.16868 of 2026-10-01: the callee talked over the
greeting, the deferred barge-in stopped its playback at two seconds, and the
greeting loop kept putting the rest of the synthesized audio on a queue nobody
drained any more. The queue holds five seconds; a ``[short pause]`` tag gave
Fish S2-Pro more than that to say, the put blocked, and everything that comes
after the greeting in the pipeline runner (the recognizer path, the dialog
worker, the watchdog's ready mark) never ran. Five caller utterances went to a
recognizer that was never read, no check-in and no stall timer fired, and the
call ended at the duration cap as 600 seconds of silence.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.core.no_input_watchdog import NoInputPolicy, NoInputWatchdog
from src.engine import Engine
from src.pipelines.base import TTSComponent
from tests.test_pipeline_runner_lifecycle import _RecordingLLM, _ResultStreamingStubSTT, _StubResolution

FRAME = b"\x01" * 320  # one 20 ms frame of slin at 8 kHz


class _LongGreetingTTS(TTSComponent):
    """Streams far more audio than the playback queue holds."""

    def __init__(self, frames=600):
        self.frames = frames
        self.yielded = 0
        self.closed = asyncio.Event()

    async def synthesize(self, call_id, text, options):
        try:
            for _ in range(self.frames):
                self.yielded += 1
                yield FRAME
                await asyncio.sleep(0)
        finally:
            self.closed.set()


class _HangingAfterFirstFrameTTS(TTSComponent):
    """Sends one frame, then stays open without ever finishing."""

    def __init__(self):
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    async def synthesize(self, call_id, text, options):
        try:
            yield FRAME
            await self.release.wait()
        finally:
            self.closed.set()


def _config():
    return AppConfig(
        **{
            "default_provider": "local",
            "providers": {"local": {"enabled": True}},
            "asterisk": {"host": "127.0.0.1", "port": 8088, "username": "u", "password": "p", "app_name": "ai-voice-agent"},
            "llm": {"initial_greeting": " [short pause]  [clears throat] Алло.", "prompt": "You are helpful", "model": "gpt-4o"},
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "audiosocket",
            "downstream_mode": "stream",
        }
    )


async def _engine_with(monkeypatch, tts, *, stream_active):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    stt = _ResultStreamingStubSTT()
    resolution = _StubResolution(stt_adapter=stt, stt_options={"streaming": True, "chunk_ms": 80}, llm_adapter=_RecordingLLM(), tts_adapter=tts)
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    manager = engine.streaming_playback_manager
    manager.start_streaming_playback = AsyncMock(return_value="stream:greeting")
    manager.stop_streaming_playback = AsyncMock(return_value=True)
    manager.is_stream_active = lambda call_id, stream_id=None: stream_active()
    engine._no_input_mark_ready = AsyncMock()
    return engine, stt


@pytest.mark.asyncio
async def test_a_greeting_cut_by_the_caller_lets_the_call_go_on(monkeypatch):
    tts = _LongGreetingTTS(frames=600)  # 12 s of audio for a 5 s queue
    # The stream lives for the first ten frames, then a barge-in has stopped it.
    stream_active = lambda: tts.yielded <= 10
    engine, stt = await _engine_with(monkeypatch, tts, stream_active=stream_active)
    call_id = "call-greeting-cut"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    try:
        await engine._ensure_pipeline_runner(session, forced=True)
        # The recognizer path starts: the greeting loop did not block the runner.
        await asyncio.wait_for(stt.started.wait(), timeout=3)
        await asyncio.wait_for(tts.closed.wait(), timeout=3)   # the synthesis was closed, not left running
        assert tts.yielded < 600
        engine._no_input_mark_ready.assert_awaited_once_with(call_id)
        [entry] = [m for m in session.conversation_history if m.get("role") == "assistant"]
        assert entry["interrupted"] is True
        assert entry["content"].strip().endswith("Алло.")
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_a_greeting_whose_synthesis_never_ends_is_cut_at_the_timeout(monkeypatch):
    tts = _HangingAfterFirstFrameTTS()
    engine, stt = await _engine_with(monkeypatch, tts, stream_active=lambda: True)
    monkeypatch.setattr("src.engine.PIPELINE_GREETING_TIMEOUT_SEC", 0.2)
    call_id = "call-greeting-timeout"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    try:
        await engine._ensure_pipeline_runner(session, forced=True)
        await asyncio.wait_for(stt.started.wait(), timeout=3)
        await asyncio.wait_for(tts.closed.wait(), timeout=3)
        engine._no_input_mark_ready.assert_awaited_once_with(call_id)
        engine.streaming_playback_manager.stop_streaming_playback.assert_awaited()
        [entry] = [m for m in session.conversation_history if m.get("role") == "assistant"]
        assert entry["interrupted"] is True
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_a_complete_greeting_is_recorded_as_before(monkeypatch):
    tts = _LongGreetingTTS(frames=20)
    engine, stt = await _engine_with(monkeypatch, tts, stream_active=lambda: True)
    call_id = "call-greeting-complete"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    try:
        await engine._ensure_pipeline_runner(session, forced=True)
        await asyncio.wait_for(stt.started.wait(), timeout=3)
        assert tts.yielded == 20
        [entry] = [m for m in session.conversation_history if m.get("role") == "assistant"]
        assert "interrupted" not in entry
        assert entry["content"].strip() == "[short pause]  [clears throat] Алло."
    finally:
        await engine._cleanup_call(call_id)


# --- the stall timer does not wait for a setup that never completes ----------------


async def _wait_until(predicate, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_the_stall_timer_ends_a_call_whose_setup_never_completes():
    hangups = []

    async def announce(call_id, message, purpose):
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, grace_timeout_sec=10, stall_timeout_sec=0.06)
    await watchdog.register("stuck-setup", policy, is_outbound=True)
    try:
        # Never marked ready: the greeting loop hung. The stall timer still counts.
        await _wait_until(lambda: hangups == ["stuck-setup"])
        assert watchdog.snapshot("stuck-setup")["phase"] == "stall_hangup"
    finally:
        await watchdog.stop("stuck-setup")


@pytest.mark.asyncio
async def test_the_greeting_playing_still_pauses_the_stall_timer_before_ready():
    hangups = []

    async def announce(call_id, message, purpose):
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, grace_timeout_sec=10, stall_timeout_sec=0.06)
    await watchdog.register("greeting-playing", policy, is_outbound=True)
    try:
        await watchdog.note_agent_output_start("greeting-playing")
        await asyncio.sleep(0.15)
        assert hangups == []  # agent audio pauses it, ready or not
        await watchdog.note_agent_output_end("greeting-playing")
        await _wait_until(lambda: hangups == ["greeting-playing"])
    finally:
        await watchdog.stop("greeting-playing")
