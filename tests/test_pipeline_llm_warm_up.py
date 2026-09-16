"""The engine warms the LLM for a call while the greeting plays.

The runner resolves the system prompt, the tools and the greeting before the
caller says a word. An adapter that offers ``warm_up`` and has it enabled
gets one call with exactly what the first turn will send: the session
history with the greeting appended, and the turn's LLM options. The task is
fire-and-forget, cancelled at hangup, and its failure never reaches the call.
"""

import asyncio
import itertools

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent
from tests.test_pipeline_runner_lifecycle import _StubResolution

_CALL_IDS = itertools.count(1)
GREETING = "Здравствуйте! Это Анна из компании Домео."
PROMPT = "Ты Анна, менеджер компании Домео."


class _WarmableLLM(LLMComponent):
    warm_up_enabled = True

    def __init__(self, *, hold=False, fail=False):
        self.warm_ups = []
        self.warmed = asyncio.Event()
        self.cancelled = asyncio.Event()
        self._hold = hold
        self._fail = fail

    async def generate(self, call_id, transcript, context, options):
        return "ok"

    async def warm_up(self, call_id, context, options):
        self.warm_ups.append((call_id, context, options))
        self.warmed.set()
        if self._fail:
            raise RuntimeError("endpoint down")
        if self._hold:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return {"status": "ok"}


class _ColdLLM(_WarmableLLM):
    warm_up_enabled = False


def _config(greeting):
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
            "llm": {"initial_greeting": greeting, "prompt": PROMPT, "model": "gpt-4o"},
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "externalmedia",
        }
    )


async def _start_call(monkeypatch, llm, *, greeting=GREETING, history=None):
    engine = Engine(_config(greeting))
    engine.pipeline_orchestrator._started = True
    resolution = _StubResolution(llm_adapter=llm)
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    # The engine remembers cleaned-up call ids for a while, so each test gets its own.
    call_id = f"call-warm-up-{next(_CALL_IDS)}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    if history:
        session.conversation_history = list(history)
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    return engine, call_id


@pytest.mark.asyncio
async def test_warm_up_gets_the_prompt_and_the_greeting_the_first_turn_will_send(monkeypatch):
    llm = _WarmableLLM()
    engine, call_id = await _start_call(monkeypatch, llm)
    try:
        await asyncio.wait_for(llm.warmed.wait(), timeout=2)
        (seen_call_id, context, options), = llm.warm_ups
        assert seen_call_id == call_id
        assert context == {"prior_messages": [{"role": "assistant", "content": GREETING}]}
        assert options["system_prompt"] == PROMPT
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_warm_up_keeps_earlier_history_ahead_of_the_greeting(monkeypatch):
    llm = _WarmableLLM()
    earlier = [{"role": "user", "content": "раньше", "timestamp": "2026-09-16T10:00:00Z"}]
    engine, call_id = await _start_call(monkeypatch, llm, history=earlier)
    try:
        await asyncio.wait_for(llm.warmed.wait(), timeout=2)
        context = llm.warm_ups[0][1]
        # Timestamps are stripped as they are for the turn; the greeting comes last.
        assert context["prior_messages"] == [
            {"role": "user", "content": "раньше"},
            {"role": "assistant", "content": GREETING},
        ]
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_warm_up_without_a_greeting_sends_the_prompt_alone(monkeypatch):
    llm = _WarmableLLM()
    engine, call_id = await _start_call(monkeypatch, llm, greeting="")
    try:
        await asyncio.wait_for(llm.warmed.wait(), timeout=2)
        assert llm.warm_ups[0][1] == {"prior_messages": []}
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_an_adapter_with_warm_up_off_is_left_alone(monkeypatch):
    llm = _ColdLLM()
    engine, call_id = await _start_call(monkeypatch, llm)
    try:
        await asyncio.sleep(0.1)
        assert llm.warm_ups == []
        assert call_id not in engine._pipeline_llm_warm_ups
        assert call_id in engine._pipeline_tasks
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_hangup_cancels_a_warm_up_still_in_flight(monkeypatch):
    llm = _WarmableLLM(hold=True)
    engine, call_id = await _start_call(monkeypatch, llm)
    await asyncio.wait_for(llm.warmed.wait(), timeout=2)
    assert call_id in engine._pipeline_llm_warm_ups

    await engine._cleanup_call(call_id)

    await asyncio.wait_for(llm.cancelled.wait(), timeout=2)
    assert call_id not in engine._pipeline_llm_warm_ups


@pytest.mark.asyncio
async def test_a_failing_warm_up_does_not_touch_the_call(monkeypatch):
    llm = _WarmableLLM(fail=True)
    engine, call_id = await _start_call(monkeypatch, llm)
    try:
        await asyncio.wait_for(llm.warmed.wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert call_id not in engine._pipeline_llm_warm_ups
        runner = engine._pipeline_tasks[call_id]
        assert not runner.done()
    finally:
        await engine._cleanup_call(call_id)
