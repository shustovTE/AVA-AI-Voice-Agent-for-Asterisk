from copy import deepcopy
from types import SimpleNamespace

from src.core.models import CallSession
from src.engine import Engine


def _generation(global_config):
    def for_agent(_policy):
        return SimpleNamespace(
            config=deepcopy(global_config),
            policy="inherit",
            requested_destination_keys=(),
            effective_destination_keys=(),
            stale_destination_keys=(),
            policies={"transfer": "inherit"},
            effective_resource_keys={"transfer": ()},
            stale_resource_keys={},
        )

    return SimpleNamespace(
        generation_id=7,
        config_hash="global-hash",
        registry=SimpleNamespace(),
        for_agent=for_agent,
    )


def _resolve(agent_hangup_policy):
    engine = Engine.__new__(Engine)
    engine._tool_generation = _generation(
        {
            "tools": {
                "hangup_call": {
                    "policy": {"markers": {"end_call": ["goodbye"]}}
                }
            }
        }
    )
    session = CallSession(call_id="call-1", caller_channel_id="call-1")
    context = SimpleNamespace(
        tool_configs=None,
        hangup_policy=agent_hangup_policy,
        in_call_http_tools=None,
    )
    Engine._resolve_session_tool_runtime(engine, session, context)
    return session


def test_agent_extend_markers_are_captured_in_call_snapshot():
    session = _resolve({"strategy": "extend", "end_call": ["да", "нет"]})
    markers = session.tool_runtime_config["tools"]["hangup_call"]["policy"][
        "markers"
    ]["end_call"]
    assert markers == ["goodbye", "да", "нет"]
    assert session.hangup_marker_policy["source"] == "agent_extend"
    assert session.hangup_marker_policy["count"] == 3


def test_agent_replace_markers_excludes_global_values():
    session = _resolve({"strategy": "replace", "end_call": ["до свидания"]})
    markers = session.tool_runtime_config["tools"]["hangup_call"]["policy"][
        "markers"
    ]["end_call"]
    assert markers == ["до свидания"]
    assert session.hangup_marker_policy["source"] == "agent_replace"


def test_agent_without_override_inherits_global_markers():
    session = _resolve(None)
    markers = session.tool_runtime_config["tools"]["hangup_call"]["policy"][
        "markers"
    ]["end_call"]
    assert markers == ["goodbye"]
    assert session.hangup_marker_policy["source"] == "global"


def _farewell_policy(flag=True, markers=None):
    return {
        "mode": "normal",
        "hangup_on_assistant_farewell": flag,
        "markers": {"assistant_farewell": markers or ["до свидания"]},
    }


def _farewell_engine(policy):
    from unittest.mock import AsyncMock, Mock

    engine = Engine.__new__(Engine)
    engine._tool_config_for_session = lambda session: {
        "tools": {"hangup_call": {"policy": policy}}
    }
    engine.session_store = SimpleNamespace(upsert_call=AsyncMock())
    engine._schedule_terminal_fallback = Mock()
    return engine


import pytest


@pytest.mark.asyncio
async def test_assistant_farewell_marker_hangs_up_after_audio():
    engine = _farewell_engine(_farewell_policy())
    session = SimpleNamespace(cleanup_after_tts=False)

    await engine._maybe_hangup_on_assistant_farewell(
        "call-1", session, "Понял вас, извините за беспокойство. До свидания."
    )

    assert session.cleanup_after_tts is True
    engine.session_store.upsert_call.assert_awaited_once_with(session)
    engine._schedule_terminal_fallback.assert_called_once()
    kwargs = engine._schedule_terminal_fallback.call_args.kwargs
    assert kwargs["reason"] == "assistant_farewell_marker"
    assert kwargs["call_outcome"] == "agent_hangup"


@pytest.mark.asyncio
async def test_assistant_farewell_requires_policy_opt_in():
    engine = _farewell_engine(_farewell_policy(flag=False))
    session = SimpleNamespace(cleanup_after_tts=False)

    await engine._maybe_hangup_on_assistant_farewell(
        "call-1", session, "До свидания."
    )

    assert session.cleanup_after_tts is False
    engine._schedule_terminal_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_assistant_farewell_ignores_mid_sentence_mentions():
    engine = _farewell_engine(_farewell_policy())
    session = SimpleNamespace(cleanup_after_tts=False)

    await engine._maybe_hangup_on_assistant_farewell(
        "call-1",
        session,
        "До свидания скажу в самом конце, а пока давайте обсудим смету и сроки ремонта",
    )

    assert session.cleanup_after_tts is False
    engine._schedule_terminal_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_assistant_farewell_does_not_double_schedule():
    engine = _farewell_engine(_farewell_policy())
    session = SimpleNamespace(cleanup_after_tts=True)

    await engine._maybe_hangup_on_assistant_farewell(
        "call-1", session, "До свидания."
    )

    engine._schedule_terminal_fallback.assert_not_called()
    engine.session_store.upsert_call.assert_not_awaited()


def test_pipeline_farewell_without_tool_honors_assistant_farewell_flag():
    policy = _farewell_policy()

    # Flag on: the agent's own farewell is sufficient, no caller end intent.
    assert Engine._is_pipeline_farewell_without_tool(
        "что?",
        "Хорошо, тогда всего доброго. До свидания.",
        policy,
    )

    # Flag off: the legacy pairing rule still applies.
    assert not Engine._is_pipeline_farewell_without_tool(
        "что?",
        "Хорошо, тогда всего доброго. До свидания.",
        _farewell_policy(flag=False),
    )
