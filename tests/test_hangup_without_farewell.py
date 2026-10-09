"""hangup_call without a farewell_message parameter.

With ``tools.hangup_call.farewell_message_enabled: false`` the tool has no
parameter: the LLM says goodbye in its reply, the tool only marks the call,
and the pipeline ends it once that reply has been heard rather than
speaking a farewell of its own.
"""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent, LLMResponse, TTSComponent
from src.tools.registry import ToolRegistry
from src.tools.runtime_config import ToolRuntimeGeneration
from src.tools.telephony import hangup as hangup_module
from src.tools.telephony.hangup import HangupCallTool
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _StubResolution

_CALL_IDS = itertools.count(1)


def _registry(hangup_config):
    registry = ToolRegistry.isolated()
    registry.register(HangupCallTool)
    registry.configure_tools({"hangup_call": hangup_config})
    return registry


# --- the definition ---------------------------------------------------------------


def test_disabling_the_farewell_removes_the_parameter():
    definition = _registry({"farewell_message_enabled": False}).get("hangup_call").definition

    assert definition.parameters == []
    assert definition.description == hangup_module.DESCRIPTION_WITHOUT_FAREWELL
    schema = definition.to_openai_schema()
    assert schema["function"]["parameters"]["properties"] == {}
    assert schema["function"]["parameters"].get("required", []) == []
    # Everything else is the same tool.
    assert definition.name == "hangup_call"
    assert definition.requires_channel is True


def test_the_parameter_stays_by_default_and_when_enabled():
    for config in ({}, {"farewell_message_enabled": True}, {"farewell_message_enabled": "yes"}):
        definition = _registry(config).get("hangup_call").definition
        assert [p.name for p in definition.parameters] == ["farewell_message"]
        assert definition.description == hangup_module.DESCRIPTION


def test_text_values_are_understood():
    for value in ("false", "0", "no", "off", False):
        assert _registry({"farewell_message_enabled": value}).get("hangup_call").farewell_message_enabled is False
    assert _registry({"farewell_message_enabled": "maybe"}).get("hangup_call").farewell_message_enabled is True


def test_the_configured_description_still_wins():
    generation = ToolRuntimeGeneration.build(
        generation_id=1,
        config={
            "tools": {
                "hangup_call": {
                    "farewell_message_enabled": False,
                    "description": "Заверши звонок после того, как попрощался в ответе.",
                }
            }
        },
    )

    definition = generation.registry.get("hangup_call").definition

    assert definition.description == "Заверши звонок после того, как попрощался в ответе."
    assert definition.parameters == []


# --- execution --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_without_a_farewell_only_marks_the_call():
    tool = _registry({"farewell_message_enabled": False}).get("hangup_call")
    context = SimpleNamespace(
        call_id="call-1",
        update_session=AsyncMock(),
        get_config_value=lambda key, default=None: "Thank you for calling. Goodbye!",
    )

    # A farewell the model passes anyway is ignored, and the default is not used.
    result = await tool.execute({"farewell_message": "До свидания!"}, context)

    assert result == {"status": "success", "message": "", "farewell_message": "", "will_hangup": True}
    context.update_session.assert_awaited_once_with(cleanup_after_tts=True)


@pytest.mark.asyncio
async def test_execution_with_the_farewell_is_unchanged():
    tool = _registry({}).get("hangup_call")
    context = SimpleNamespace(
        call_id="call-1",
        update_session=AsyncMock(),
        get_config_value=lambda key, default=None: "Thank you for calling. Goodbye!",
    )

    assert (await tool.execute({"farewell_message": "До свидания!"}, context))["message"] == "До свидания!"
    assert (await tool.execute({}, context))["message"] == "Thank you for calling. Goodbye!"


# --- the pipeline -----------------------------------------------------------------


class _GoodbyeLLM(LLMComponent):
    """Says goodbye in its reply and asks for the hangup in the same turn."""

    def __init__(self, farewell=None):
        self.done = asyncio.Event()
        self._farewell = farewell

    async def generate(self, call_id, transcript, context, options):
        self.done.set()
        parameters = {"farewell_message": self._farewell} if self._farewell else {}
        return LLMResponse(
            text="Хорошо, всего доброго!",
            tool_calls=[{"id": "hangup-1", "name": "hangup_call", "parameters": parameters}],
        )


class _TextRecordingTTS(TTSComponent):
    def __init__(self):
        self.texts = []

    async def synthesize(self, call_id, text, options):
        self.texts.append(text)
        if False:
            yield b""


def _config():
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
            "pipelines": {"goodbye": {}},
            "active_pipeline": "goodbye",
            "audio_transport": "audiosocket",
        }
    )


async def _goodbye_turn(monkeypatch, hangup_config, *, farewell=None):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    stt = _ResultStreamingStubSTT()
    llm = _GoodbyeLLM(farewell)
    tts = _TextRecordingTTS()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=tts,
    )
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    monkeypatch.setattr(
        engine.transport_orchestrator,
        "get_context_config",
        lambda *a, **k: SimpleNamespace(
            prompt=None,
            greeting=None,
            tools=["hangup_call"],
            in_call_http_tools={},
            disable_global_in_call_tools=[],
        ),
    )
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    terminations = []

    async def _terminate(call_id, **kwargs):
        terminations.append((call_id, kwargs))
        return True

    monkeypatch.setattr(engine, "_terminate_call_after_audio", _terminate)

    call_id = f"call-goodbye-{next(_CALL_IDS)}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "goodbye"
    session.context_name = "goodbye-context"
    session.allowed_tools = ["hangup_call"]
    session.tool_runtime_registry = _registry(hangup_config)
    await engine.session_store.upsert_call(session)

    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    await stt.results.put("bye")
    await asyncio.wait_for(llm.done.wait(), timeout=2)
    for _ in range(100):
        if terminations:
            break
        await asyncio.sleep(0.02)
    return engine, call_id, tts, terminations


@pytest.mark.asyncio
async def test_the_pipeline_ends_the_call_after_the_reply_with_no_farewell_of_its_own(monkeypatch):
    engine, call_id, tts, terminations = await _goodbye_turn(monkeypatch, {"farewell_message_enabled": False})
    try:
        # Only the reply was synthesized; nothing was added after it.
        assert tts.texts == ["Хорошо, всего доброго!"]
        assert terminations == [(call_id, {"reason": "pipeline_hangup_call", "audio_already_drained": False})]
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_the_pipeline_still_speaks_the_farewell_when_the_parameter_is_on(monkeypatch):
    engine, call_id, tts, terminations = await _goodbye_turn(monkeypatch, {}, farewell="До свидания!")
    try:
        assert tts.texts == ["Хорошо, всего доброго!", "До свидания!"]
        assert terminations == [(call_id, {"reason": "pipeline_hangup_call", "audio_already_drained": True})]
    finally:
        await engine._cleanup_call(call_id)
