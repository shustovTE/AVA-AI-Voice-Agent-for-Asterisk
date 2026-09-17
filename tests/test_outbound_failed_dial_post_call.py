"""Post-call tools for outbound attempts that never became a call.

Post-call webhooks ran only from ``_cleanup_call`` with a CallSession, so a
dial that was rejected, rang out, met a busy line, an answering machine or
a declined consent was never reported: a CRM learned the outcome of the
calls the agent had and nothing about the rest. Every finished attempt is
now reported once, through the same post-call tools and with the same
payload fields; ``call_outcome`` names how the dial ended, ``error_message``
why, the transcript is empty and the duration 0.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import CallSession
from src.engine import Engine
from src.tools.base import PostCallTool, ToolCategory, ToolDefinition, ToolPhase
from src.tools.context import PostCallContext
from src.tools.http.generic_webhook import GenericWebhookTool, WebhookConfig, create_webhook_tool
from src.tools.registry import ToolRegistry


class _RecordingTool(PostCallTool):
    """A global post-call tool that keeps every context it was run with."""

    def __init__(self, name, *, on_failed_dial=True):
        self.contexts = []
        self._on_failed_dial = on_failed_dial
        self._definition = ToolDefinition(
            name=name,
            description=name,
            category=ToolCategory.BUSINESS,
            phase=ToolPhase.POST_CALL,
            is_global=True,
            timeout_ms=1000,
        )

    @property
    def definition(self):
        return self._definition

    def runs_on_failed_dial(self):
        return self._on_failed_dial

    async def execute(self, context):
        self.contexts.append(context)


class _PlainTool(_RecordingTool):
    """A post-call tool that never heard of failed dials (the base default)."""

    def __init__(self, name):
        super().__init__(name)
        del self._on_failed_dial

    runs_on_failed_dial = PostCallTool.runs_on_failed_dial


def _meta(**overrides):
    meta = {
        "attempt_id": "attempt-1",
        "campaign_id": "campaign-1",
        "lead_id": "lead-1",
        "phone_number": "+15551230001",
        "context": "sales",
        "routing_method": "ai_agent",
        "provider": None,
        "lead_name": "Иван",
        "custom_vars": {"amo_lead_id": "4242"},
        "created_at_ts": time.time() - 5,
    }
    meta.update(overrides)
    return meta


def _engine(*tools):
    registry = ToolRegistry.isolated()
    for tool in tools:
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
    engine._outbound_extension_identity = "6789"
    engine._outbound_pbx_type = "freepbx"
    engine._outbound_dial_context = "from-internal"
    engine._outbound_dial_prefix = ""
    engine._outbound_channel_tech = "local_only"
    engine._outbound_agent_selector = lambda campaign, lead: ("sales", "ai_agent")
    engine._outbound_routing_channel_vars = lambda context_name, routing_method: {"AI_AGENT": context_name}
    engine._outbound_build_amd_opts = lambda options: ""
    engine._outbound_attempt_meta_by_attempt_id = {}
    engine._outbound_attempt_meta_by_channel_id = {}
    engine._outbound_attempt_amd = {}
    engine._outbound_awaiting_amd_channel_ids = set()
    engine._seen_outbound_channels = set()
    engine._seen_caller_stasis_channels = set()
    engine.session_store = SimpleNamespace(get_by_channel_id=AsyncMock(return_value=None))
    engine.outbound_store = SimpleNamespace(
        finish_attempt=AsyncMock(),
        set_lead_state=AsyncMock(),
        set_attempt_channel=AsyncMock(),
        set_attempt_gate_result=AsyncMock(),
        get_campaign=AsyncMock(return_value={"voicemail_drop_enabled": 1}),
    )
    engine.ari_client = SimpleNamespace(
        originate_channel=AsyncMock(return_value={"id": "chan-1"}),
        hangup_channel=AsyncMock(return_value=True),
        send_command=AsyncMock(return_value={"status": 404}),
        play_media=AsyncMock(return_value={"id": "playback-1"}),
        set_channel_var=AsyncMock(),
    )
    engine._wait_for_ari_playback = AsyncMock()
    engine._set_outbound_agent_channel_vars = AsyncMock()

    engine.fired = []

    def fire(coro, *, name=None):
        task = asyncio.ensure_future(coro)
        engine.fired.append(task)
        return task

    engine._fire_and_forget = fire
    return engine


async def _settle(engine):
    if engine.fired:
        await asyncio.gather(*engine.fired)


def _seed(engine, meta):
    engine._outbound_attempt_meta_by_attempt_id[meta["attempt_id"]] = meta
    if meta.get("channel_id"):
        engine._outbound_attempt_meta_by_channel_id[meta["channel_id"]] = meta
    return meta


def _only_context(tool):
    assert len(tool.contexts) == 1, [c.call_outcome for c in tool.contexts]
    return tool.contexts[0]


# --- the dial never produced a channel --------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_originate_is_reported_with_the_attempts_fields():
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    _seed(engine, _meta())
    engine.ari_client.originate_channel = AsyncMock(return_value={"status": 500, "reason": "Allocation failed"})

    await engine._outbound_originate_attempt(
        {"id": "campaign-1", "voicemail_drop_enabled": 1, "consent_enabled": 0},
        {"id": "lead-1", "phone_number": "+15551230001", "name": "Иван"},
        "attempt-1",
    )
    await _settle(engine)

    context = _only_context(tool)
    assert context.call_id == "attempt-1"  # no channel was ever created
    assert context.attempt_id == "attempt-1"
    assert context.call_outcome == "error"
    assert context.error_message == "Allocation failed"
    assert context.call_direction == "outbound"
    assert context.caller_number == "+15551230001"
    assert context.called_number == "+15551230001"
    assert context.caller_name == "Иван"
    assert context.campaign_id == "campaign-1"
    assert context.lead_id == "lead-1"
    assert context.custom_vars == {"amo_lead_id": "4242"}
    assert context.context_name == "sales"
    assert context.provider == "openai_realtime"
    assert context.call_duration_seconds == 0
    assert context.conversation_history == []
    assert context.summary is None
    assert context.call_start_time and context.call_end_time
    assert context.config == {"tools": {}}
    engine.outbound_store.finish_attempt.assert_awaited_once_with(
        "attempt-1", outcome="error", error_message="Allocation failed"
    )
    assert engine._outbound_attempt_meta_by_attempt_id == {}


@pytest.mark.asyncio
async def test_a_lead_that_could_not_be_marked_dialing_is_reported_as_canceled():
    tool = _RecordingTool("crm")
    engine = _engine(tool)

    await engine._outbound_post_call_tools_for_attempt(
        _meta(), outcome="canceled", error_message="Lead not leased (state transition failed)"
    )
    await _settle(engine)

    context = _only_context(tool)
    assert (context.call_outcome, context.error_message) == (
        "canceled",
        "Lead not leased (state transition failed)",
    )


# --- the channel died before the answer -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cause_txt, outcome",
    [
        ("User busy", "busy"),
        ("No answer", "no_answer"),
        ("Unknown", "no_answer"),
        ("Congestion", "congestion"),
        ("Channel unavailable", "chanunavail"),
        ("Call Rejected", "error"),
    ],
)
async def test_a_channel_destroyed_before_the_answer_reports_the_hangup_cause(cause_txt, outcome):
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    _seed(engine, _meta(channel_id="chan-1", originated_at_ts=time.time() - 30))

    await engine._handle_outbound_channel_destroyed(
        {"type": "ChannelDestroyed", "channel": {"id": "chan-1"}, "cause": 17, "cause_txt": cause_txt}
    )
    await _settle(engine)

    context = _only_context(tool)
    assert context.call_id == "chan-1"
    assert context.attempt_id == "attempt-1"
    assert context.call_outcome == outcome
    assert context.error_message == cause_txt
    assert context.caller_number == "+15551230001"
    assert context.lead_id == "lead-1"
    assert engine.outbound_store.finish_attempt.await_args.kwargs["outcome"] == outcome
    assert "chan-1" in engine._seen_outbound_channels
    assert engine._outbound_attempt_meta_by_channel_id == {}
    assert engine._outbound_attempt_meta_by_attempt_id == {}


@pytest.mark.asyncio
async def test_a_stale_attempt_is_reported_as_no_answer():
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    _seed(engine, _meta(channel_id="chan-1", originated_at_ts=time.time() - 100_000))

    await engine._outbound_cleanup_stale_attempts()
    await _settle(engine)

    context = _only_context(tool)
    assert context.call_id == "chan-1"
    assert context.call_outcome == "no_answer"
    assert context.error_message == "stale originate (no StasisStart)"
    assert engine._outbound_attempt_meta_by_attempt_id == {}


# --- the call was answered but never reached the agent ----------------------------


@pytest.mark.asyncio
async def test_an_answering_machine_is_reported_after_the_voicemail_drop():
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    _seed(engine, _meta(channel_id="chan-1", originated_at_ts=time.time() - 20))

    await engine._handle_outbound_amd_result(
        "chan-1", {}, ["outbound_amd", "attempt-1", "MACHINE", "LONGGREETING", "", ""]
    )
    await _settle(engine)

    context = _only_context(tool)
    assert context.call_id == "chan-1"
    assert context.call_outcome == "voicemail_dropped"
    assert context.error_message is None
    assert context.caller_name == "Иван"
    engine.ari_client.play_media.assert_awaited_once()
    engine.ari_client.hangup_channel.assert_awaited_once_with("chan-1")
    assert engine._outbound_attempt_meta_by_attempt_id == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("consent, outcome", [("denied", "consent_denied"), ("timeout", "consent_timeout")])
async def test_a_declined_consent_is_reported(consent, outcome):
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    _seed(engine, _meta(channel_id="chan-1"))

    await engine._handle_outbound_amd_result(
        "chan-1", {}, ["outbound_amd", "attempt-1", "HUMAN", "", "2" if consent == "denied" else "", consent]
    )
    await _settle(engine)

    context = _only_context(tool)
    assert context.call_outcome == outcome
    assert context.call_id == "chan-1"
    engine.ari_client.hangup_channel.assert_awaited_once_with("chan-1")


@pytest.mark.asyncio
async def test_a_call_rejected_before_startup_is_reported_once():
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    meta = _seed(engine, _meta(channel_id="chan-1"))

    await engine._reject_outbound_answered_attempt("chan-1", "attempt-1", meta, "unsafe answered metadata")
    await _settle(engine)
    context = _only_context(tool)
    assert (context.call_id, context.call_outcome, context.error_message) == (
        "chan-1",
        "error",
        "unsafe answered metadata",
    )

    # The ChannelDestroyed that follows the hangup finds the fail-closed marker
    # (or nothing at all) and does not report the attempt again.
    await engine._handle_outbound_channel_destroyed(
        {"type": "ChannelDestroyed", "channel": {"id": "chan-1"}, "cause": 16, "cause_txt": "Normal Clearing"}
    )
    await _settle(engine)
    assert len(tool.contexts) == 1


# --- who is reported to, and when not ---------------------------------------------


@pytest.mark.asyncio
async def test_only_tools_that_opt_in_run():
    wants = _RecordingTool("crm")
    declines = _RecordingTool("summary_mailer", on_failed_dial=False)
    plain = _PlainTool("legacy")
    engine = _engine(wants, declines, plain)
    _seed(engine, _meta(channel_id="chan-1"))

    await engine._handle_outbound_channel_destroyed(
        {"type": "ChannelDestroyed", "channel": {"id": "chan-1"}, "cause_txt": "User busy"}
    )
    await _settle(engine)

    assert len(wants.contexts) == 1
    assert declines.contexts == []
    assert plain.contexts == []


@pytest.mark.asyncio
async def test_the_agents_context_decides_which_tools_run():
    global_tool = _RecordingTool("everyone")
    sales_tool = _RecordingTool("sales_only")
    sales_tool._definition.is_global = False
    engine = _engine(global_tool, sales_tool)
    engine.transport_orchestrator.get_context_config = lambda name, routing_method=None: SimpleNamespace(
        post_call_tools=["sales_only"] if name == "sales" else [],
        disable_global_post_call_tools=["everyone"] if name == "sales" else [],
    )

    await engine._outbound_post_call_tools_for_attempt(_meta(context="sales"), outcome="busy")
    await engine._outbound_post_call_tools_for_attempt(_meta(attempt_id="attempt-2", context="support"), outcome="busy")
    await _settle(engine)

    assert [c.attempt_id for c in sales_tool.contexts] == ["attempt-1"]
    assert [c.attempt_id for c in global_tool.contexts] == ["attempt-2"]


@pytest.mark.asyncio
async def test_a_channel_with_a_session_is_left_to_the_calls_own_cleanup():
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    _seed(engine, _meta(channel_id="chan-1"))
    engine.session_store.get_by_channel_id = AsyncMock(return_value=SimpleNamespace(is_outbound=True))

    await engine._handle_outbound_channel_destroyed(
        {"type": "ChannelDestroyed", "channel": {"id": "chan-1"}, "cause_txt": "Normal Clearing"}
    )
    await _settle(engine)

    assert tool.contexts == []
    engine.outbound_store.finish_attempt.assert_not_awaited()


@pytest.mark.asyncio
async def test_nothing_is_reported_without_attempt_metadata_and_a_failing_tool_does_not_break_the_dialer():
    tool = _RecordingTool("crm")
    engine = _engine(tool)

    await engine._outbound_post_call_tools_for_attempt(None, outcome="busy")
    await engine._outbound_post_call_tools_for_attempt({"channel_id": "chan-1"}, outcome="busy")
    assert tool.contexts == []

    engine._tool_generation = SimpleNamespace(registry=None, config={})  # selection blows up
    await engine._outbound_post_call_tools_for_attempt(_meta(), outcome="busy")  # logged, not raised


# --- the same fields, the same webhooks -------------------------------------------


def _session_context(engine, monkeypatch):
    """The context a real outbound call hands to the same tools."""
    monkeypatch.setattr("src.core.call_history.get_call_history_store", lambda: None)
    session = CallSession(call_id="chan-9", caller_channel_id="chan-9")
    session.caller_number = "+15551230001"
    session.called_number = "+15551230001"
    session.caller_name = "Иван"
    session.context_name = "sales"
    session.provider_name = "openai_realtime"
    session.is_outbound = True
    session.start_time = datetime.now(timezone.utc)
    session.conversation_history = [{"role": "user", "content": "Алло"}]
    session.outbound_campaign_id = "campaign-1"
    session.outbound_lead_id = "lead-1"
    session.outbound_attempt_id = "attempt-9"
    session.outbound_custom_vars = {"amo_lead_id": "4242"}
    session.error_message = "provider startup failed"
    return session


@pytest.mark.asyncio
async def test_a_failed_dial_carries_exactly_the_fields_of_a_call(monkeypatch):
    tool = _RecordingTool("crm")
    engine = _engine(tool)
    session = _session_context(engine, monkeypatch)

    await engine._execute_post_call_tools("chan-9", session, call_duration_seconds=12, call_outcome="error")
    await engine._outbound_post_call_tools_for_attempt(_meta(channel_id="chan-1"), outcome="busy", error_message="User busy")
    await _settle(engine)

    from_call, from_dial = tool.contexts
    assert set(from_dial.to_payload_dict()) == set(from_call.to_payload_dict())
    # The session path now names its attempt and its error too.
    assert from_call.attempt_id == "attempt-9"
    assert from_call.error_message == "provider startup failed"
    assert from_call.to_payload_dict()["attempt_id"] == "attempt-9"
    assert from_dial.to_payload_dict()["attempt_id"] == "attempt-1"
    assert from_dial.to_payload_dict()["error_message"] == "User busy"
    assert from_dial.to_payload_dict()["transcript_json"] == "[]"
    assert from_dial.to_payload_dict()["call_duration"] == 0
    assert from_dial.to_payload_dict()["amo_lead_id"] == "4242"


@pytest.mark.asyncio
async def test_an_existing_webhook_receives_the_failed_dial_through_its_own_template():
    webhook = create_webhook_tool(
        "crm_webhook",
        {
            "is_global": True,
            "url": "https://crm.example.com/calls/{attempt_id}",
            "payload_template": (
                '{"event_type": "call_completed", "call_id": "{call_id}", "outcome": "{call_outcome}", '
                '"error": "{error_message}", "attempt_id": "{attempt_id}", "lead_id": "{lead_id}", '
                '"amo_lead_id": "{amo_lead_id}", "phone": "{caller_number}", "name": "{caller_name}", '
                '"duration": {call_duration}, "transcript": {transcript_json}, "summary": {summary_json}}'
            ),
        },
    )
    engine = _engine(webhook)
    _seed(engine, _meta(channel_id="chan-1", lead_name=""))

    response = AsyncMock(status=200)
    response.text = AsyncMock(return_value="ok")
    request_cm = AsyncMock()
    request_cm.__aenter__ = AsyncMock(return_value=response)
    request_cm.__aexit__ = AsyncMock(return_value=None)
    http = AsyncMock()
    http.request = MagicMock(return_value=request_cm)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=http)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=session_cm):
        await engine._handle_outbound_channel_destroyed(
            {"type": "ChannelDestroyed", "channel": {"id": "chan-1"}, "cause": 17, "cause_txt": "User busy"}
        )
        await _settle(engine)

    kwargs = http.request.call_args.kwargs
    assert kwargs["url"] == "https://crm.example.com/calls/attempt-1"
    assert json.loads(kwargs["data"]) == {
        "event_type": "call_completed",
        "call_id": "chan-1",
        "outcome": "busy",
        "error": "User busy",
        "attempt_id": "attempt-1",
        "lead_id": "lead-1",
        "amo_lead_id": "4242",
        "phone": "+15551230001",
        "name": "Outbound +15551230001",
        "duration": 0,
        "transcript": [],
        "summary": "",
    }
    # Diagnostics were recorded and consumed; nothing was written to call history.
    assert webhook.get_last_result(call_id="chan-1") is None


# --- the webhook switch -----------------------------------------------------------


def test_webhooks_report_failed_dials_unless_switched_off():
    assert WebhookConfig(name="x").send_on_failed_dial is True
    assert create_webhook_tool("x", {}).runs_on_failed_dial() is True
    for value in (False, "false", "0", "no", "off"):
        assert create_webhook_tool("x", {"send_on_failed_dial": value}).runs_on_failed_dial() is False, value
    for value in (True, "true", "1", "yes", "on", None, ""):
        assert create_webhook_tool("x", {"send_on_failed_dial": value}).runs_on_failed_dial() is True, value
    assert GenericWebhookTool(WebhookConfig(name="x", send_on_failed_dial=False)).runs_on_failed_dial() is False
    assert _PlainTool("legacy").runs_on_failed_dial() is False


def test_the_payload_names_the_attempt_and_the_error_in_url_headers_and_body():
    context = PostCallContext(
        call_id="chan-1",
        caller_number="+15551230001",
        call_outcome="busy",
        attempt_id="attempt-1",
        error_message="User busy",
        custom_vars={"error_message": "never wins over the built-in", "amo_lead_id": "4242"},
    )
    payload = context.to_payload_dict()
    assert payload["attempt_id"] == "attempt-1"
    assert payload["error_message"] == "User busy"
    assert payload["amo_lead_id"] == "4242"

    tool = GenericWebhookTool(WebhookConfig(name="x", url="https://crm.example.com/{attempt_id}"))
    assert tool._substitute_variables("https://crm.example.com/{attempt_id}/{lead_id}", context) == (
        "https://crm.example.com/attempt-1/"
    )
    empty = PostCallContext(call_id="c", caller_number="1").to_payload_dict()
    assert (empty["attempt_id"], empty["error_message"]) == ("", "")
