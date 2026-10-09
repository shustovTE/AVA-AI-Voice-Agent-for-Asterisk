"""Caller-inactivity announcements in a modular pipeline.

The check-in and the final message are synthesized by the pipeline's TTS adapter. A
pipeline whose replies stream over the call's media (AudioSocket / ExternalMedia) must
deliver them the same way: the file player needs the media directory on the Asterisk
host, which a split-server deployment does not have, and it takes 8 kHz mu-law only.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.engine import Engine


class _StreamingStub:
    def __init__(self):
        self.stream_id = "no-input-stream"
        self.active = True
        self.start_args = None
        self.queue = None
        self.drained = []

    def is_stream_active(self, call_id, stream_id=None):
        return self.active and stream_id == self.stream_id

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.start_args = (call_id, kwargs)
        self.queue = queue
        return self.stream_id

    async def stop_streaming_playback(self, call_id, *, drain=False):
        if drain:
            while True:
                chunk = await self.queue.get()
                if chunk is None:
                    break
                self.drained.append(chunk)
        self.active = False
        return True


class _TTS:
    downstream_mode_override = "auto"

    def __init__(self):
        self.spoken = []

    async def synthesize(self, call_id, text, options):
        self.spoken.append(text)
        yield b"ulaw-1"
        yield b"ulaw-2"


class _FilePlayer:
    def __init__(self):
        self.played = []

    async def play_audio(self, call_id, audio_bytes, playback_type):
        self.played.append((call_id, audio_bytes, playback_type))
        return "playback-1"

    async def wait_for_playback_end(self, call_id, playback_id, timeout_sec=None):
        return True


class _NoFilePlayer:
    async def play_audio(self, *args, **kwargs):
        raise AssertionError("a streaming pipeline must not fall back to file playback")


def _engine(downstream_mode: str, playback_manager):
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(downstream_mode=downstream_mode)
    session = SimpleNamespace(
        call_id="call-1",
        cleanup_in_progress=False,
        pipeline_name="ru",
        no_input_state={},
        conversation_history=[],
    )
    engine.session_store = SimpleNamespace(get_by_call_id=AsyncMock(return_value=session))
    engine._save_session = AsyncMock()
    engine._session_was_transferred = lambda s: False
    engine._begin_provider_output_operation = lambda *a, **k: None
    engine._end_provider_output_operation = lambda *a, **k: None
    tts = _TTS()
    pipeline = SimpleNamespace(tts_adapter=tts, tts_options={"format": {"encoding": "mulaw", "sample_rate": 8000}})
    engine.pipeline_orchestrator = SimpleNamespace(get_pipeline=lambda call_id, name: pipeline)
    engine.playback_manager = playback_manager
    engine.streaming_playback_manager = _StreamingStub()
    return engine, session, tts


@pytest.mark.asyncio
async def test_a_streaming_pipeline_speaks_the_check_in_on_the_call_media_stream():
    engine, session, tts = _engine("stream", _NoFilePlayer())

    assert await engine._speak_no_input_announcement("call-1", "Вы ещё здесь?", "check_in") is True

    manager = engine.streaming_playback_manager
    assert tts.spoken == ["Вы ещё здесь?"]
    assert manager.start_args == (
        "call-1",
        {"playback_type": "no-input-check_in", "source_encoding": "mulaw", "source_sample_rate": 8000},
    )
    assert manager.drained == [b"ulaw-1", b"ulaw-2"]
    assert session.conversation_history[-1]["content"] == "Вы ещё здесь?"
    assert session.conversation_history[-1]["event"] == "no_input_check_in"
    assert session.no_input_state["announcement_delivery_complete"] is True
    assert session.no_input_state["announcement_active"] is False


@pytest.mark.asyncio
async def test_a_file_playback_pipeline_keeps_the_file_player():
    player = _FilePlayer()
    engine, session, tts = _engine("file", player)

    assert await engine._speak_no_input_announcement("call-1", "До свидания.", "final") is True

    assert player.played == [("call-1", b"ulaw-1ulaw-2", "no-input-final")]
    assert engine.streaming_playback_manager.start_args is None
    assert session.conversation_history[-1]["event"] == "no_input_final"


@pytest.mark.asyncio
async def test_a_failed_stream_reports_the_announcement_as_not_spoken():
    engine, session, tts = _engine("stream", _NoFilePlayer())

    async def broken(*args, **kwargs):
        raise RuntimeError("stream refused")

    engine.streaming_playback_manager.start_streaming_playback = broken

    assert await engine._speak_no_input_announcement("call-1", "Вы ещё здесь?", "check_in") is False
    assert session.no_input_state["announcement_delivery_complete"] is False
    assert session.conversation_history == []
