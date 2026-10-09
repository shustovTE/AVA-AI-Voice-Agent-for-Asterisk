"""A tool's result is answered with the same prompt and tools as the turn.

After a tool runs, the pipeline feeds its result back to the model and speaks
the follow-up. That request must carry what the first one carried: the
agent's system prompt, the tool allowlist and the pipeline's LLM options,
or the follow-up is spoken out of persona and cannot call a second tool.
"""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent, LLMResponse
from src.tools.base import Tool, ToolCategory, ToolDefinition
from src.tools.registry import ToolRegistry
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _SilentTTS, _StubResolution

_CALL_IDS = itertools.count(1)
AGENT_PROMPT = "Ты Анна, менеджер компании Домео."
TOOL_MESSAGE = "Retrieved: Available: True, Next Slot: 10:00"


class _RecordingToolLLM(LLMComponent):
    """Asks for a tool on the first request and answers on the second."""

    def __init__(self):
        self.calls = []  # (transcript, context, options)
        self.completed = asyncio.Event()

    async def generate(self, call_id, transcript, context, options):
        self.calls.append((transcript, context, dict(options or {})))
        if len(self.calls) == 1:
            return LLMResponse(
                text="Секунду, проверяю.",
                tool_calls=[{"id": "tool-1", "name": "lookup_tool", "parameters": {"date": "2026-09-17"}}],
            )
        self.completed.set()
        return LLMResponse(text="Есть окно в десять.", tool_calls=[])


class _LookupTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(name="lookup_tool", description="Test lookup", category=ToolCategory.BUSINESS)

    async def execute(self, parameters, context):
        return {"status": "success", "data": {"available": True, "next_slot": "10:00"}, "message": TOOL_MESSAGE}


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
            "llm": {"initial_greeting": "", "prompt": "Global prompt", "model": "gpt-4o"},
            "pipelines": {"tools": {}},
            "active_pipeline": "tools",
            "audio_transport": "audiosocket",
        }
    )


async def _turn_with_a_tool(monkeypatch):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    stt = _ResultStreamingStubSTT()
    llm = _RecordingToolLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=_SilentTTS(),
    )
    resolution.llm_options = {"max_tokens": 120}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    monkeypatch.setattr(
        engine.transport_orchestrator,
        "get_context_config",
        lambda *a, **k: SimpleNamespace(
            prompt=AGENT_PROMPT,
            greeting=None,
            tools=["lookup_tool"],
            in_call_http_tools={},
            disable_global_in_call_tools=[],
        ),
    )
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    call_id = f"call-tool-continuation-{next(_CALL_IDS)}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "tools"
    session.context_name = "tool-context"
    session.allowed_tools = ["lookup_tool"]
    registry = ToolRegistry.isolated()
    registry.register_instance(_LookupTool())
    session.tool_runtime_registry = registry
    await engine.session_store.upsert_call(session)

    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    await stt.results.put("проверь завтра")
    await asyncio.wait_for(llm.completed.wait(), timeout=2)
    await asyncio.sleep(0.05)
    return engine, call_id, llm


@pytest.mark.asyncio
async def test_the_follow_up_request_carries_the_agent_prompt_and_the_tools(monkeypatch):
    engine, call_id, llm = await _turn_with_a_tool(monkeypatch)
    try:
        assert len(llm.calls) == 2
        first, second = llm.calls[0][2], llm.calls[1][2]
        assert first["system_prompt"] == AGENT_PROMPT
        assert first["tools"] == ["lookup_tool"]
        assert first["max_tokens"] == 120
        # The follow-up is the same conversation: same persona, same tools, same options.
        assert second["system_prompt"] == AGENT_PROMPT
        assert second["tools"] == ["lookup_tool"]
        assert second["max_tokens"] == 120
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_the_follow_up_request_sees_the_tool_result(monkeypatch):
    engine, call_id, llm = await _turn_with_a_tool(monkeypatch)
    try:
        transcript, context, _options = llm.calls[1]
        assert transcript == ""
        messages = context["prior_messages"]
        assert messages[-1] == {"role": "tool", "content": TOOL_MESSAGE, "tool_call_id": "tool-1"}
        assert messages[-2]["role"] == "assistant"
        assert messages[-2]["tool_calls"][0]["function"]["name"] == "lookup_tool"
        assert messages[-3] == {"role": "assistant", "content": "Секунду, проверяю."}
        assert messages[-4] == {"role": "user", "content": "проверь завтра"}
    finally:
        await engine._cleanup_call(call_id)
