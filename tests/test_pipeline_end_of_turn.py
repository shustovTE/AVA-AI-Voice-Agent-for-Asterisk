"""The dialog worker waits for the caller to stop before answering.

Streaming STT returns a result at every phrase boundary. Answering the first one
talked over callers working through a long sentence, and TTS playback muted the
capture path so the rest of the sentence was lost outright.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from tests.test_pipeline_runner_lifecycle import (
    _RecordingLLM,
    _ResultStreamingStubSTT,
    _StubResolution,
)


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
            "llm": {
                "initial_greeting": "",
                "prompt": "You are helpful",
                "model": "gpt-4o",
            },
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "externalmedia",
        }
    )


async def _start_call(monkeypatch, call_id, llm_options):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    stt = _ResultStreamingStubSTT()
    llm = _RecordingLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
    )
    resolution.llm_options = dict(llm_options)
    monkeypatch.setattr(
        engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution
    )
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, stt, llm


@pytest.mark.asyncio
async def test_results_within_the_silence_window_become_one_turn(monkeypatch):
    """The caller gets to finish the sentence instead of being answered twice."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-debounce-merge",
        {"end_of_turn_silence_ms": 400},
    )
    try:
        await stt.results.put("я хочу")
        await asyncio.sleep(0.1)
        await stt.results.put("сделать перепланировку")

        await asyncio.wait_for(llm.called.wait(), timeout=3)
        assert llm.transcripts == ["я хочу сделать перепланировку"]
    finally:
        await engine._cleanup_call("call-debounce-merge")


@pytest.mark.asyncio
async def test_retired_thresholds_no_longer_shorten_the_turn(monkeypatch):
    """The reported production config is now answered on silence, not length."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-debounce-legacy",
        {
            "aggregation_timeout_sec": 0.4,
            "aggregation_min_words": 1,
            "aggregation_min_chars": 8,
        },
    )
    try:
        await stt.results.put("я хочу")
        await asyncio.sleep(0.1)
        await stt.results.put("сделать перепланировку")

        await asyncio.wait_for(llm.called.wait(), timeout=3)
        assert llm.transcripts == ["я хочу сделать перепланировку"]
    finally:
        await engine._cleanup_call("call-debounce-legacy")


@pytest.mark.asyncio
async def test_a_one_word_answer_is_not_left_waiting(monkeypatch):
    """Silence ends the turn, so a bare "yes" is as prompt as a paragraph."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-debounce-single",
        {"end_of_turn_silence_ms": 200},
    )
    try:
        await stt.results.put("да")

        await asyncio.wait_for(llm.called.wait(), timeout=3)
        assert llm.transcripts == ["да"]
    finally:
        await engine._cleanup_call("call-debounce-single")


@pytest.mark.asyncio
async def test_empty_results_do_not_push_the_deadline_out(monkeypatch):
    """Silence reported as an empty result is not speech and must not extend."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-debounce-empty",
        {"end_of_turn_silence_ms": 300},
    )
    try:
        await stt.results.put("перепланировка")
        for _ in range(12):
            await stt.results.put("")
            await asyncio.sleep(0.05)

        await asyncio.wait_for(llm.called.wait(), timeout=3)
        assert llm.transcripts == ["перепланировка"]
    finally:
        await engine._cleanup_call("call-debounce-empty")


@pytest.mark.asyncio
async def test_max_wait_releases_a_caller_who_never_pauses(monkeypatch):
    """The optional cap bounds the debounce for an unbroken monologue."""
    engine, stt, llm = await _start_call(
        monkeypatch,
        "call-debounce-cap",
        {"end_of_turn_silence_ms": 30000, "end_of_turn_max_wait_ms": 500},
    )

    async def keep_talking():
        for index in range(40):
            await stt.results.put(f"слово{index}")
            await asyncio.sleep(0.05)

    talker = asyncio.create_task(keep_talking())
    try:
        # Without the cap this would sit on the 30 s silence window.
        await asyncio.wait_for(llm.called.wait(), timeout=3)
        assert llm.transcripts[0].startswith("слово0 ")
    finally:
        talker.cancel()
        await engine._cleanup_call("call-debounce-cap")
