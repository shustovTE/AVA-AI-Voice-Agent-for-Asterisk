"""An outbound session carries its attempt, lead, Agent and pipeline from the first save.

Reproduces call 1790771032.16030 of 2026-09-30: the callee answered, AMD said
HUMAN, the far end dropped 40 ms after the session was saved. Cleanup found the
session and ran the post-call webhook with what the session held at that
moment: the default full-agent provider, no Agent, no lead, custom_vars
placeholders left as ``{amo_lead_id}``. The attempt's own data had been read
from channel variables after the save, one ARI round trip each, and every read
returned 404 on the dead channel; the attempt stayed open and was reported a
second time two minutes later as ``no_answer``. The engine holds all of it in
memory since the originate, so the session is seeded from there before it is
first saved, and channel variables remain the fallback for an engine restarted
mid-call.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.engine as engine_module
from src.core.models import CallSession
from src.core.session_store import SessionStore
from src.engine import Engine
from src.tools.base import PostCallTool, ToolCategory, ToolDefinition, ToolPhase
from src.tools.registry import ToolRegistry

CHANNEL = "1790771032.16030"
PHONE = "+79688143204"
PIPELINE = "T-One-Gemme-eleven"


class _Stop(Exception):
    """Ends the setup handler right after the save under test."""


class _RecordingTool(PostCallTool):
    def __init__(self):
        self.contexts = []
        self._definition = ToolDefinition(
            name="crm", description="crm", category=ToolCategory.BUSINESS,
            phase=ToolPhase.POST_CALL, is_global=True, timeout_ms=1000,
        )

    @property
    def definition(self):
        return self._definition

    def runs_on_failed_dial(self):
        return True

    async def execute(self, context):
        self.contexts.append(context)


@pytest.fixture(autouse=True)
def _clean_module_guards():
    engine_module._cleanup_in_progress.clear()
    engine_module._cleanup_completed_at.clear()
    yield
    engine_module._cleanup_in_progress.clear()
    engine_module._cleanup_completed_at.clear()


def _meta(**overrides):
    meta = {
        "attempt_id": "attempt-1",
        "campaign_id": "campaign-1",
        "lead_id": "lead-1",
        "phone_number": PHONE,
        "context": "receptionist",
        "routing_method": "ai_agent",
        "provider": None,
        "lead_name": "Иван",
        "custom_vars": {"amo_lead_id": "4242", "client_name": "Иван"},
        "channel_id": CHANNEL,
    }
    meta.update(overrides)
    return meta


def _context_config(name, routing_method=None):
    if name == "receptionist":
        return SimpleNamespace(pipeline=PIPELINE, post_call_tools=[], disable_global_post_call_tools=[], provider=None)
    return None


def _engine(tool=None, *, meta=None):
    registry = ToolRegistry.isolated()
    if tool is not None:
        registry.register_instance(tool)
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(default_provider="deepgram", asterisk=SimpleNamespace(app_name="ai-voice-agent"))
    engine.providers = {}
    engine.provider_kinds = {}
    engine._vad_mode = "auto"
    engine.vad_manager = None
    engine._tool_generation = SimpleNamespace(registry=registry, config={"tools": {}}) if tool is not None else None
    engine.transport_orchestrator = SimpleNamespace(
        get_context_config=_context_config,
        agent_store=SimpleNamespace(default_slug=lambda: "default_agent"),
    )
    engine.session_store = SessionStore()
    engine.bridges = {}
    engine._destroyed_channel_ts = {}
    engine._orphan_first_seen = {}
    engine._called_number_cache = {}
    engine._outbound_extension_identity = "99999"
    engine._outbound_attempt_meta_by_channel_id = {}
    engine._outbound_attempt_meta_by_attempt_id = {}
    engine._outbound_attempt_amd = {}
    engine._seen_outbound_channels = set()
    engine._seen_caller_stasis_channels = set()
    if meta is not None:
        engine._outbound_attempt_meta_by_channel_id[CHANNEL] = meta
        engine._outbound_attempt_meta_by_attempt_id[meta["attempt_id"]] = meta
    engine.outbound_store = SimpleNamespace(
        finish_attempt=AsyncMock(),
        set_lead_state=AsyncMock(),
        get_active_attempt_runtime_context=AsyncMock(return_value=None),
    )
    engine.ari_client = SimpleNamespace(
        send_command=AsyncMock(return_value={"status": 404, "reason": "Provided channel was not found"}),
        answer_channel=AsyncMock(return_value=True),
        create_bridge=AsyncMock(return_value="bridge-1"),
        add_channel_to_bridge=AsyncMock(return_value=True),
        destroy_bridge=AsyncMock(return_value=True),
        hangup_channel=AsyncMock(return_value=True),
    )
    engine._cleanup_call = AsyncMock()
    engine._set_outbound_agent_channel_vars = AsyncMock()
    engine.fired = []

    def fire(coro, *, name=None):
        task = asyncio.ensure_future(coro)
        engine.fired.append(task)
        return task

    engine._fire_and_forget = fire
    return engine


def _channel():
    return {"id": CHANNEL, "name": f"Local/79688143204@from-internal-000011b4;1",
            "caller": {"name": "Новая сделка по входящему звонку из UIS", "number": PHONE}}


def _variables_read(engine):
    return [c.kwargs.get("params", {}).get("variable") for c in engine.ari_client.send_command.await_args_list]


# --- the session is complete from its first save ------------------------------------


async def test_the_first_save_carries_the_attempt_the_lead_the_agent_and_the_pipeline():
    engine = _engine(meta=_meta())
    engine._save_session = AsyncMock(side_effect=[None, _Stop()])  # stop after the called-number step

    await engine._handle_caller_stasis_start_hybrid(CHANNEL, _channel())

    session = engine._save_session.await_args_list[0].args[0]
    assert session.is_outbound is True
    assert (session.outbound_attempt_id, session.outbound_campaign_id, session.outbound_lead_id) == (
        "attempt-1", "campaign-1", "lead-1")
    assert (session.caller_number, session.called_number, session.caller_name) == (PHONE, PHONE, "Иван")
    assert session.outbound_custom_vars == {"amo_lead_id": "4242", "client_name": "Иван"}
    assert (session.context_name, session.routing_method) == ("receptionist", "ai_agent")
    assert (session.provider_name, session.pipeline_name) == ("pipeline", PIPELINE)
    # The channel was already gone: AAVA_OUTBOUND could not be read, the attempt in
    # memory still made the call outbound, and nothing else was read back from it.
    engine.ari_client.answer_channel.assert_not_awaited()
    assert _variables_read(engine) == ["AAVA_OUTBOUND"]
    assert session.called_number == PHONE  # not replaced by "unknown"
    engine._cleanup_call.assert_awaited_once()


async def test_channel_variables_remain_the_fallback_after_an_engine_restart():
    engine = _engine()  # no attempt metadata in memory

    async def send_command(method, path, params=None, tolerate_statuses=None, **_):
        var = (params or {}).get("variable")
        values = {"AAVA_OUTBOUND": "1", "AAVA_ATTEMPT_ID": "attempt-9", "AAVA_CAMPAIGN_ID": "campaign-9",
                  "AAVA_LEAD_ID": "lead-9", "AAVA_OUTBOUND_PHONE": PHONE}
        return {"value": values.get(var, "")}

    engine.ari_client.send_command = AsyncMock(side_effect=send_command)
    engine.outbound_store.get_active_attempt_runtime_context = AsyncMock(return_value={"custom_vars": {"amo_lead_id": "9"}})
    engine._save_session = AsyncMock(side_effect=[None, None, _Stop()])  # creation, called number, outbound vars

    await engine._handle_caller_stasis_start_hybrid(CHANNEL, _channel())

    session = engine._save_session.await_args_list[0].args[0]
    assert session.is_outbound is True
    assert session.outbound_attempt_id == "attempt-9"
    assert session.outbound_campaign_id == "campaign-9"
    assert session.outbound_lead_id == "lead-9"
    assert session.called_number == PHONE
    assert session.outbound_custom_vars == {"amo_lead_id": "9"}
    assert "AAVA_ATTEMPT_ID" in _variables_read(engine)


def test_seeding_names_a_lead_without_a_name_by_its_number_and_copies_the_custom_vars():
    engine = _engine()
    session = CallSession(call_id=CHANNEL, caller_channel_id=CHANNEL, caller_name="99999", caller_number="99999")
    meta = _meta(lead_name="", attempt_id="")
    assert engine._seed_outbound_session_from_attempt(session, meta) is False  # no attempt id to link
    assert session.caller_name == f"Outbound {PHONE}"
    assert session.outbound_custom_vars == meta["custom_vars"]
    assert session.outbound_custom_vars is not meta["custom_vars"]
    assert engine._preassign_context_pipeline(session) == PIPELINE
    assert (session.provider_name, session.provider_kind) == ("pipeline", "pipeline")


def test_no_pipeline_is_preassigned_for_an_agent_without_one():
    engine = _engine()
    engine.transport_orchestrator.get_context_config = lambda name, routing_method=None: SimpleNamespace(pipeline=None)
    session = CallSession(call_id=CHANNEL, caller_channel_id=CHANNEL, provider_name="deepgram", context_name="sales")
    assert engine._preassign_context_pipeline(session) is None
    assert session.provider_name == "deepgram"


# --- the resolver keeps the seeded Agent when the channel has no variables -----------


@pytest.mark.parametrize(
    "channel_vars, session_kwargs, expected",
    [
        ({}, dict(is_outbound=True, outbound_attempt_id="attempt-1", context_name="receptionist", routing_method="ai_agent"),
         ("receptionist", "ai_agent")),
        ({"AI_AGENT": "sales"}, dict(is_outbound=True, outbound_attempt_id="attempt-1", context_name="receptionist"),
         ("sales", "ai_agent")),
        ({}, dict(is_outbound=False, context_name="receptionist"), ("default_agent", "default")),
        ({}, dict(is_outbound=True, outbound_attempt_id=None, context_name="receptionist"), ("default_agent", "default")),
    ],
)
def test_the_agent_of_a_call_comes_from_the_channel_then_the_attempt_then_the_default(channel_vars, session_kwargs, expected):
    engine = _engine()
    session = CallSession(call_id=CHANNEL, caller_channel_id=CHANNEL)
    for key, value in session_kwargs.items():
        setattr(session, key, value)
    assert Engine._resolve_context_from_channel_vars(engine, session, channel_vars) == expected


def test_without_a_default_agent_the_resolver_returns_nothing():
    engine = _engine()
    engine.transport_orchestrator.agent_store = SimpleNamespace(default_slug=lambda: None)
    session = CallSession(call_id=CHANNEL, caller_channel_id=CHANNEL)
    assert Engine._resolve_context_from_channel_vars(engine, session, {}) == (None, None)


# --- what a cleanup racing setup now reports ---------------------------------------


async def test_a_cleanup_racing_setup_reports_the_lead_and_closes_the_attempt(monkeypatch):
    tool = _RecordingTool()
    engine = _engine(tool, meta=_meta())
    session = CallSession(call_id=CHANNEL, caller_channel_id=CHANNEL, caller_name="Новая сделка",
                          caller_number=PHONE, provider_name="deepgram")
    engine._seed_outbound_session_from_attempt(session, _meta())
    engine._preassign_context_pipeline(session)
    engine.transport_orchestrator.get_context_config = lambda name, routing_method=None: SimpleNamespace(
        post_call_tools=[], disable_global_post_call_tools=[], provider=None)
    engine.transport_orchestrator.agent_store = SimpleNamespace(default_slug=lambda: "default_agent")

    saved = []

    class _Store:
        _enabled = True

        async def save(self, record):
            saved.append(record)
            return True

        async def get_by_call_id(self, call_id):
            return next((r for r in saved if r.call_id == call_id), None)

    monkeypatch.setattr("src.core.call_history.get_call_history_store", lambda: _Store())

    await engine._execute_post_call_tools(CHANNEL, session, call_duration_seconds=0, call_outcome="caller_hangup")
    if engine.fired:
        await asyncio.gather(*engine.fired)
    await engine._persist_call_history(session, CHANNEL)

    [context] = tool.contexts
    assert (context.campaign_id, context.lead_id, context.attempt_id) == ("campaign-1", "lead-1", "attempt-1")
    assert context.custom_vars == {"amo_lead_id": "4242", "client_name": "Иван"}
    assert (context.context_name, context.provider, context.call_direction) == ("receptionist", "pipeline", "outbound")
    assert context.to_payload_dict()["amo_lead_id"] == "4242"

    [record] = saved
    assert (record.caller_number, record.called_number, record.context_name) == (PHONE, PHONE, "receptionist")
    assert (record.provider_name, record.pipeline_name, record.outcome) == ("pipeline", PIPELINE, "abandoned")
    engine.outbound_store.finish_attempt.assert_awaited_once()
    kwargs = engine.outbound_store.finish_attempt.await_args.kwargs
    assert kwargs["call_history_call_id"] == record.id
    assert kwargs["outcome"] == "answered_human"
    # The attempt is closed in memory too: the stale-attempt watchdog has nothing to report later.
    assert engine._outbound_attempt_meta_by_attempt_id == {}
    assert engine._outbound_attempt_meta_by_channel_id == {}
