"""Call history for an outbound call that the far end dropped right after answering.

Reproduces call 1790688333.14314 of 2026-09-29: the callee answered, AMD said
HUMAN, and the operator's side hung up before the agent session existed. The
attempt was finished with its data and the post-call webhook carried the lead,
the campaign and the hangup cause, but Call History showed an empty row
("Unknown", 0 s, "Call ended before session registration") written by the
no-session cleanup that StasisEnd triggered before the attempt was finalized.
The row must carry the attempt's data, the attempt must link to it, and the
setup that still ran for the dead channel must not leak a bridge.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.engine as engine_module
from src.core.session_store import SessionStore
from src.engine import Engine
from src.tools.base import PostCallTool, ToolCategory, ToolDefinition, ToolPhase
from src.tools.registry import ToolRegistry

CHANNEL = "1790688333.14314"
PHONE = "+15551230001"


class _FakeHistoryStore:
    _enabled = True

    def __init__(self):
        self.records = []

    async def save(self, record):
        if any(r.call_id == record.call_id for r in self.records):
            return True  # dedupe-by-call_id, as the real store
        self.records.append(record)
        return True

    async def get_by_call_id(self, call_id):
        return next((r for r in self.records if r.call_id == call_id), None)


class _RecordingTool(PostCallTool):
    def __init__(self):
        self.contexts = []
        self._definition = ToolDefinition(
            name="crm",
            description="crm",
            category=ToolCategory.BUSINESS,
            phase=ToolPhase.POST_CALL,
            is_global=True,
            timeout_ms=1000,
        )

    @property
    def definition(self):
        return self._definition

    def runs_on_failed_dial(self):
        return True

    async def execute(self, context):
        self.contexts.append(context)


@pytest.fixture
def history(monkeypatch):
    store = _FakeHistoryStore()
    monkeypatch.setattr("src.core.call_history.get_call_history_store", lambda: store)
    return store


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
        "context": "sales",
        "routing_method": "ai_agent",
        "provider": None,
        "lead_name": "Иван",
        "custom_vars": {"amo_lead_id": "4242"},
        "created_at_ts": time.time() - 30,
        "originated_at_ts": time.time() - 25,
    }
    meta.update(overrides)
    return meta


def _engine(tool=None):
    registry = ToolRegistry.isolated()
    if tool is not None:
        registry.register_instance(tool)

    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(
        default_provider="openai_realtime",
        asterisk=SimpleNamespace(app_name="ai-voice-agent"),
    )
    engine._tool_generation = SimpleNamespace(registry=registry, config={"tools": {}})
    engine.transport_orchestrator = SimpleNamespace(
        get_context_config=lambda name, routing_method=None: SimpleNamespace(
            post_call_tools=[], disable_global_post_call_tools=[], provider=None
        )
    )
    engine.providers = {}
    engine.bridges = {}
    engine.session_store = SessionStore()
    engine._attended_transfer_agent_channel_to_call_id = {}
    engine._seen_aux_channels = set()
    engine._seen_outbound_channels = set()
    engine._seen_caller_stasis_channels = set()
    engine._outbound_attempt_meta_by_attempt_id = {}
    engine._outbound_attempt_meta_by_channel_id = {}
    engine._outbound_attempt_amd = {}
    engine._outbound_awaiting_amd_channel_ids = set()
    engine._outbound_forced_hangup_tasks = {}
    engine._outbound_amd_context = "aava-outbound-amd"
    engine._outbound_build_amd_opts = lambda options: ""
    engine._set_outbound_agent_channel_vars = AsyncMock()
    engine._destroyed_channel_ts = {}
    engine._orphan_first_seen = {}
    engine.outbound_store = SimpleNamespace(
        finish_attempt=AsyncMock(),
        set_lead_state=AsyncMock(),
        set_attempt_channel=AsyncMock(),
        set_attempt_gate_result=AsyncMock(),
        get_campaign=AsyncMock(return_value={"voicemail_drop_enabled": 1}),
        get_active_attempt_runtime_context=AsyncMock(return_value=None),
    )
    engine.ari_client = SimpleNamespace(
        send_command=AsyncMock(return_value={"status": 404}),
        hangup_channel=AsyncMock(return_value=True),
        set_channel_var=AsyncMock(return_value=True),
        continue_in_dialplan=AsyncMock(return_value=True),
        answer_channel=AsyncMock(return_value=True),
        create_bridge=AsyncMock(return_value="bridge-1"),
        add_channel_to_bridge=AsyncMock(return_value=True),
        destroy_bridge=AsyncMock(return_value=True),
    )
    engine.fired = []

    def fire(coro, *, name=None):
        task = asyncio.ensure_future(coro)
        engine.fired.append(task)
        return task

    engine._fire_and_forget = fire
    return engine


def _seed(engine, meta):
    engine._outbound_attempt_meta_by_attempt_id[meta["attempt_id"]] = meta
    if meta.get("channel_id"):
        engine._outbound_attempt_meta_by_channel_id[meta["channel_id"]] = meta
    return meta


async def _settle(engine):
    if engine.fired:
        await asyncio.gather(*engine.fired)


def _destroyed(channel_id=CHANNEL, cause_txt="Normal Clearing", cause=16):
    return {"type": "ChannelDestroyed", "channel": {"id": channel_id}, "cause": cause, "cause_txt": cause_txt}


# --- the history row ------------------------------------------------------------------


async def test_a_call_dropped_after_the_answer_gets_a_history_row_with_the_attempts_data(history):
    tool = _RecordingTool()
    engine = _engine(tool)
    _seed(engine, _meta(channel_id=CHANNEL, answered_at_ts=time.time() - 4))
    engine._outbound_attempt_amd["attempt-1"] = {
        "amd_status": "HUMAN",
        "amd_cause": "TOOLONG-0",
        "consent_dtmf": None,
        "consent_result": None,
    }

    await engine._handle_outbound_channel_destroyed(_destroyed())
    await _settle(engine)

    [record] = history.records
    assert record.call_id == CHANNEL
    assert (record.caller_number, record.called_number, record.caller_name) == (PHONE, PHONE, "Иван")
    assert record.outcome == "abandoned"
    assert record.error_message is None
    assert record.external_direction == "outbound"
    assert (record.context_name, record.routing_method, record.provider_name) == ("sales", "ai_agent", "openai_realtime")
    assert 3.5 <= record.duration_seconds <= 10
    assert record.end_time >= record.start_time
    assert record.conversation_history == []
    assert record.external_metadata == {
        "call_direction": "outbound",
        "attempt_id": "attempt-1",
        "campaign_id": "campaign-1",
        "lead_id": "lead-1",
        "attempt_outcome": "error",
        "hangup_cause": "Normal Clearing",
        "ended_before_session": True,
        "amd_status": "HUMAN",
        "amd_cause": "TOOLONG-0",
    }

    # The attempt is finished once, with its AMD verdict and hangup cause intact,
    # and linked to the row so Call Scheduling opens it.
    engine.outbound_store.finish_attempt.assert_awaited_once()
    kwargs = engine.outbound_store.finish_attempt.await_args.kwargs
    assert kwargs["call_history_call_id"] == record.id
    assert (kwargs["outcome"], kwargs["error_message"]) == ("error", "Normal Clearing")
    assert (kwargs["amd_status"], kwargs["amd_cause"]) == ("HUMAN", "TOOLONG-0")
    engine.outbound_store.set_lead_state.assert_awaited_once_with("lead-1", state="failed", last_outcome="error")

    # The webhook still fires once, as before.
    [context] = tool.contexts
    assert (context.call_id, context.call_outcome, context.error_message) == (CHANNEL, "error", "Normal Clearing")
    assert context.caller_number == PHONE
    assert engine._outbound_attempt_meta_by_channel_id == {}
    assert engine._outbound_attempt_meta_by_attempt_id == {}
    assert CHANNEL in engine._seen_outbound_channels


async def test_an_answer_without_an_amd_verdict_is_enough_for_a_row(history):
    engine = _engine()
    _seed(engine, _meta(channel_id=CHANNEL, answered_at_ts=time.time() - 2, lead_name=""))

    await engine._handle_outbound_channel_destroyed(_destroyed(cause_txt="Normal Clearing"))
    await _settle(engine)

    [record] = history.records
    assert record.caller_name == f"Outbound {PHONE}"
    assert "amd_status" not in record.external_metadata
    assert record.external_metadata["hangup_cause"] == "Normal Clearing"
    assert 1.5 <= record.duration_seconds <= 10
    assert engine.outbound_store.finish_attempt.await_args.kwargs["call_history_call_id"] == record.id


async def test_a_dial_that_was_never_answered_gets_no_history_row(history):
    engine = _engine()
    _seed(engine, _meta(channel_id=CHANNEL))

    await engine._handle_outbound_channel_destroyed(_destroyed(cause_txt="User busy", cause=17))
    await _settle(engine)

    assert history.records == []
    kwargs = engine.outbound_store.finish_attempt.await_args.kwargs
    assert kwargs["outcome"] == "busy"
    assert "call_history_call_id" not in kwargs


async def test_a_history_store_that_is_off_or_broken_does_not_stop_the_attempt(monkeypatch):
    engine = _engine()
    _seed(engine, _meta(channel_id=CHANNEL, answered_at_ts=time.time() - 1))

    disabled = _FakeHistoryStore()
    disabled._enabled = False
    monkeypatch.setattr("src.core.call_history.get_call_history_store", lambda: disabled)
    assert await engine._persist_outbound_attempt_history(
        _meta(), channel_id=CHANNEL, attempt_outcome="error", hangup_cause="Normal Clearing", amd=None
    ) is None

    broken = _FakeHistoryStore()
    broken.save = AsyncMock(side_effect=RuntimeError("disk full"))
    monkeypatch.setattr("src.core.call_history.get_call_history_store", lambda: broken)
    await engine._handle_outbound_channel_destroyed(_destroyed())
    await _settle(engine)

    kwargs = engine.outbound_store.finish_attempt.await_args.kwargs
    assert kwargs["outcome"] == "error"
    assert "call_history_call_id" not in kwargs
    assert engine._outbound_attempt_meta_by_channel_id == {}


# --- the answer is stamped -------------------------------------------------------------


async def test_the_answer_is_stamped_on_the_attempt_metadata():
    engine = _engine()
    _seed(engine, _meta())

    before = time.time()
    await engine._handle_outbound_answered(CHANNEL, {"id": CHANNEL}, ["outbound", "attempt-1"])

    meta = engine._outbound_attempt_meta_by_channel_id[CHANNEL]
    assert meta is engine._outbound_attempt_meta_by_attempt_id["attempt-1"]
    assert before <= meta["answered_at_ts"] <= time.time()
    engine.ari_client.continue_in_dialplan.assert_awaited_once()


async def test_metadata_recovered_after_a_restart_is_stamped_too():
    engine = _engine()
    engine.outbound_store.get_active_attempt_runtime_context = AsyncMock(return_value=_meta())

    await engine._handle_outbound_answered(CHANNEL, {"id": CHANNEL}, ["outbound", "attempt-1"])

    assert engine._outbound_attempt_meta_by_channel_id[CHANNEL]["answered_at_ts"] > 0


# --- no empty stub from the no-session cleanup -----------------------------------------


@pytest.mark.parametrize("order", ["stasis_end_first", "channel_destroyed_first"])
async def test_the_no_session_cleanup_leaves_the_row_to_the_attempts_finalizer(history, order):
    """StasisEnd and ChannelDestroyed each trigger a cleanup that finds no session;
    whichever runs first, exactly one row is written, and it is the attempt's."""
    engine = _engine()
    _seed(engine, _meta(channel_id=CHANNEL, answered_at_ts=time.time() - 3))
    engine._seen_caller_stasis_channels.add(CHANNEL)  # the HUMAN path marked it

    async def stasis_end_cleanup():
        await engine._cleanup_call(CHANNEL)

    async def channel_destroyed():
        engine._note_channel_destroyed(CHANNEL)
        await engine._handle_outbound_channel_destroyed(_destroyed())
        await engine._cleanup_call(CHANNEL)

    if order == "stasis_end_first":
        await stasis_end_cleanup()
        assert history.records == []  # nothing written before the attempt is finalized
        await channel_destroyed()
    else:
        await channel_destroyed()
        await stasis_end_cleanup()
    await _settle(engine)

    assert [r.call_id for r in history.records] == [CHANNEL]
    assert history.records[0].caller_number == PHONE
    assert history.records[0].error_message is None
    assert CHANNEL not in engine._seen_caller_stasis_channels
    assert CHANNEL not in engine._seen_outbound_channels


async def test_an_inbound_caller_that_hung_up_before_setup_still_gets_the_abandoned_stub(history):
    engine = _engine()
    engine._seen_caller_stasis_channels.add("PJSIP/inbound-0001")

    await engine._cleanup_call("PJSIP/inbound-0001")

    [record] = history.records
    assert (record.call_id, record.outcome) == ("PJSIP/inbound-0001", "abandoned")
    assert record.error_message == "Call ended before session registration"


# --- call setup on a dead channel -----------------------------------------------------


def _outbound_channel():
    return {"id": CHANNEL, "name": f"PJSIP/trunk-{CHANNEL}", "caller": {"name": "Иван", "number": PHONE}}


async def test_call_setup_is_skipped_for_a_channel_that_is_already_destroyed():
    engine = _engine()
    engine.ari_client.send_command = AsyncMock(return_value={"value": "1"})  # AAVA_OUTBOUND
    engine._note_channel_destroyed(CHANNEL)

    await engine._handle_caller_stasis_start_hybrid(CHANNEL, _outbound_channel())

    engine.ari_client.answer_channel.assert_not_awaited()
    engine.ari_client.create_bridge.assert_not_awaited()
    assert engine.bridges == {}
    assert await engine.session_store.get_by_call_id(CHANNEL) is None


async def test_a_setup_that_fails_on_a_dead_channel_destroys_the_bridge_it_created():
    engine = _engine()
    engine.ari_client.send_command = AsyncMock(return_value={"value": "1"})
    engine.ari_client.add_channel_to_bridge = AsyncMock(return_value=False)  # "Channel not found"
    engine._cleanup_call = AsyncMock()

    await engine._handle_caller_stasis_start_hybrid(CHANNEL, _outbound_channel())

    engine._cleanup_call.assert_awaited_once_with(CHANNEL, force_caller_hangup=True)
    engine.ari_client.destroy_bridge.assert_awaited_once_with("bridge-1")
    assert engine.bridges == {}
