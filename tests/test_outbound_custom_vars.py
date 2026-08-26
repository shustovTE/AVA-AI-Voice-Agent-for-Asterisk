import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.engine import Engine


def _engine(*, custom_vars=None):
    engine = Engine.__new__(Engine)
    engine._outbound_amd_context = "aava-outbound-amd"
    engine._outbound_awaiting_amd_channel_ids = set()
    engine._outbound_attempt_meta_by_attempt_id = {
        "attempt-1": {
            "attempt_id": "attempt-1",
            "campaign_id": "campaign-1",
            "lead_id": "lead-1",
            "context": "sales",
            "routing_method": "ai_agent",
            "custom_vars": custom_vars or {},
        }
    }
    engine._outbound_attempt_meta_by_channel_id = {
        "channel-1": engine._outbound_attempt_meta_by_attempt_id["attempt-1"]
    }
    engine._outbound_attempt_amd = {}
    engine._outbound_forced_hangup_tasks = {}
    engine._seen_outbound_channels = set()
    engine._set_outbound_agent_channel_vars = AsyncMock()
    engine.ari_client = SimpleNamespace(
        set_channel_var=AsyncMock(return_value=True),
        continue_in_dialplan=AsyncMock(return_value=True),
        send_command=AsyncMock(return_value={"status": 404}),
        hangup_channel=AsyncMock(return_value=True),
    )
    engine.outbound_store = SimpleNamespace(
        set_attempt_channel=AsyncMock(),
        set_lead_state=AsyncMock(),
        get_campaign=AsyncMock(return_value={}),
        get_active_attempt_runtime_context=AsyncMock(return_value=None),
        finish_attempt=AsyncMock(),
    )
    return engine


# ─── Answered flow: custom_vars never touch the channel ──────────────────────


@pytest.mark.asyncio
async def test_outbound_answered_never_probes_channel_for_custom_vars():
    engine = _engine(custom_vars={"task": "confirm appointment"})

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )

    # No serialize/write/read-back dance on the channel: custom_vars live in
    # attempt metadata and the durable store only.
    engine.ari_client.send_command.assert_not_awaited()
    assert not any(
        call.args[1] == "AAVA_CUSTOM_VARS_JSON"
        for call in engine.ari_client.set_channel_var.await_args_list
    )
    engine.ari_client.continue_in_dialplan.assert_awaited_once()
    engine.outbound_store.finish_attempt.assert_not_awaited()
    engine.ari_client.hangup_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_answered_channel_remap_drops_stale_originated_channel_owner():
    engine = _engine(custom_vars={"task": "confirm appointment"})
    meta = engine._outbound_attempt_meta_by_attempt_id["attempt-1"]
    meta["channel_id"] = "originated-channel"
    engine._outbound_attempt_meta_by_channel_id = {
        "originated-channel": meta,
    }

    await engine._handle_outbound_answered(
        "answered-channel",
        {"id": "answered-channel"},
        ["outbound", "attempt-1"],
    )

    assert "originated-channel" not in engine._outbound_attempt_meta_by_channel_id
    assert engine._outbound_attempt_meta_by_channel_id["answered-channel"][
        "channel_id"
    ] == "answered-channel"


@pytest.mark.asyncio
async def test_answered_call_recovers_metadata_after_in_memory_state_loss():
    custom_vars = {"task": "confirm appointment"}
    engine = _engine(custom_vars=custom_vars)
    recovered = dict(engine._outbound_attempt_meta_by_attempt_id["attempt-1"])
    engine._outbound_attempt_meta_by_attempt_id.clear()
    engine._outbound_attempt_meta_by_channel_id.clear()
    engine.outbound_store.get_active_attempt_runtime_context.return_value = recovered

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )

    engine.outbound_store.get_active_attempt_runtime_context.assert_awaited_once_with(
        "attempt-1"
    )
    engine.ari_client.continue_in_dialplan.assert_awaited_once()
    assert engine._outbound_attempt_meta_by_attempt_id["attempt-1"][
        "channel_id"
    ] == "channel-1"
    assert engine._outbound_attempt_meta_by_attempt_id["attempt-1"][
        "custom_vars"
    ] == custom_vars


@pytest.mark.asyncio
async def test_answered_call_fails_closed_when_attempt_metadata_is_unrecoverable():
    engine = _engine()
    engine._outbound_attempt_meta_by_attempt_id.clear()
    engine._outbound_attempt_meta_by_channel_id.clear()

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )

    engine.ari_client.continue_in_dialplan.assert_not_awaited()
    engine.outbound_store.finish_attempt.assert_awaited_once_with(
        "attempt-1",
        outcome="error",
        error_message="outbound attempt metadata unavailable after answer",
    )
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-1")
    assert "channel-1" in engine._seen_outbound_channels


@pytest.mark.asyncio
async def test_answered_call_fails_closed_on_corrupt_durable_custom_vars():
    engine = _engine()
    recovered = dict(engine._outbound_attempt_meta_by_attempt_id["attempt-1"])
    recovered["custom_vars_valid"] = False
    engine._outbound_attempt_meta_by_attempt_id.clear()
    engine._outbound_attempt_meta_by_channel_id.clear()
    engine.outbound_store.get_active_attempt_runtime_context.return_value = recovered

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )

    engine.ari_client.continue_in_dialplan.assert_not_awaited()
    engine.outbound_store.finish_attempt.assert_awaited_once_with(
        "attempt-1",
        outcome="error",
        error_message="outbound custom_vars metadata is invalid after answer",
    )
    engine.outbound_store.set_lead_state.assert_awaited_once_with(
        "lead-1",
        state="failed",
        last_outcome="error",
    )
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-1")


@pytest.mark.asyncio
async def test_outbound_answered_logs_amd_pending_persistence_failure():
    engine = _engine()
    engine.outbound_store.set_lead_state.side_effect = RuntimeError(
        "database unavailable"
    )

    with patch("src.engine.logger.warning") as warning:
        await engine._handle_outbound_answered(
            "channel-1",
            {"id": "channel-1"},
            ["outbound", "attempt-1"],
        )

    warning.assert_any_call(
        "Failed to persist amd_pending lead state after answer",
        channel_id="channel-1",
        attempt_id="attempt-1",
        lead_id="lead-1",
        exc_info=True,
    )
    engine.ari_client.continue_in_dialplan.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejection_remains_fail_closed_when_persistence_fails():
    engine = _engine(custom_vars={"task": "confirm appointment"})
    engine._outbound_attempt_meta_by_attempt_id["attempt-1"]["custom_vars_valid"] = False
    engine.outbound_store.finish_attempt.side_effect = RuntimeError("database unavailable")

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )

    engine.ari_client.continue_in_dialplan.assert_not_awaited()
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-1")


@pytest.mark.asyncio
async def test_rejected_outbound_hangup_retains_owner_until_retry_is_accepted(
    monkeypatch,
):
    engine = _engine(custom_vars={"task": "confirm appointment"})
    engine._outbound_attempt_meta_by_attempt_id["attempt-1"]["custom_vars_valid"] = False
    engine.ari_client.hangup_channel.side_effect = [False, False, True]
    retry_started = asyncio.Event()
    allow_retry = asyncio.Event()

    async def controlled_sleep(_seconds):
        retry_started.set()
        await allow_retry.wait()

    monkeypatch.setattr("src.engine.asyncio.sleep", controlled_sleep)

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )
    await retry_started.wait()

    assert "attempt-1" in engine._outbound_attempt_meta_by_attempt_id
    assert "channel-1" in engine._outbound_attempt_meta_by_channel_id
    assert len(engine._outbound_forced_hangup_tasks) == 1

    # Re-scheduling the same channel must retain the existing single owner.
    existing_task = engine._outbound_forced_hangup_tasks["channel-1"]
    engine._schedule_outbound_forced_hangup_retry(
        attempt_id="attempt-1",
        channel_id="channel-1",
    )
    assert engine._outbound_forced_hangup_tasks["channel-1"] is existing_task

    allow_retry.set()
    await existing_task

    assert engine.ari_client.hangup_channel.await_count == 3
    assert engine._outbound_attempt_meta_by_attempt_id == {}
    assert engine._outbound_attempt_meta_by_channel_id == {}
    assert engine._outbound_forced_hangup_tasks == {}


@pytest.mark.asyncio
async def test_destroyed_rejected_channel_cancels_retry_without_overwriting_error(
    monkeypatch,
):
    engine = _engine(custom_vars={"task": "confirm appointment"})
    engine._outbound_attempt_meta_by_attempt_id["attempt-1"]["custom_vars_valid"] = False
    engine.ari_client.hangup_channel.return_value = False
    retry_started = asyncio.Event()
    keep_retry_waiting = asyncio.Event()

    async def controlled_sleep(_seconds):
        retry_started.set()
        await keep_retry_waiting.wait()

    monkeypatch.setattr("src.engine.asyncio.sleep", controlled_sleep)

    await engine._handle_outbound_answered(
        "channel-1",
        {"id": "channel-1"},
        ["outbound", "attempt-1"],
    )
    await retry_started.wait()
    retry_task = engine._outbound_forced_hangup_tasks["channel-1"]

    await engine._handle_outbound_channel_destroyed(
        {"channel": {"id": "channel-1"}, "cause_txt": "Unknown"}
    )
    await retry_task

    engine.outbound_store.finish_attempt.assert_awaited_once_with(
        "attempt-1",
        outcome="error",
        error_message="outbound custom_vars metadata is invalid after answer",
    )
    assert engine._outbound_attempt_meta_by_attempt_id == {}
    assert engine._outbound_attempt_meta_by_channel_id == {}
    assert engine._outbound_forced_hangup_tasks == {}


# ─── Hydration: session custom_vars come from metadata, not the channel ──────


def _hydration_session(**overrides):
    defaults = dict(outbound_attempt_id="attempt-1", outbound_custom_vars=None)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_hydrate_custom_vars_uses_in_memory_attempt_metadata():
    engine = _engine(custom_vars={"task": "confirm appointment"})
    session = _hydration_session()

    await engine._hydrate_outbound_custom_vars(session)

    assert session.outbound_custom_vars == {"task": "confirm appointment"}
    engine.outbound_store.get_active_attempt_runtime_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_hydrate_custom_vars_falls_back_to_durable_store():
    engine = _engine()  # in-memory meta carries no custom_vars
    engine.outbound_store.get_active_attempt_runtime_context.return_value = {
        "custom_vars": {"task": "from the store"},
        "custom_vars_valid": True,
    }
    session = _hydration_session()

    await engine._hydrate_outbound_custom_vars(session)

    assert session.outbound_custom_vars == {"task": "from the store"}
    engine.outbound_store.get_active_attempt_runtime_context.assert_awaited_once_with(
        "attempt-1"
    )


@pytest.mark.asyncio
async def test_hydrate_custom_vars_is_noop_without_attempt_id():
    engine = _engine(custom_vars={"task": "ignored"})
    session = _hydration_session(outbound_attempt_id="")

    await engine._hydrate_outbound_custom_vars(session)

    assert session.outbound_custom_vars is None
    engine.outbound_store.get_active_attempt_runtime_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_hydrate_custom_vars_survives_store_errors():
    engine = _engine()
    engine.outbound_store.get_active_attempt_runtime_context.side_effect = RuntimeError(
        "database unavailable"
    )
    session = _hydration_session()

    await engine._hydrate_outbound_custom_vars(session)

    assert session.outbound_custom_vars is None


# ─── Prompt template substitution of custom_vars ─────────────────────────────


def _substitution_session(**overrides):
    defaults = dict(
        call_id="call-1",
        caller_name="Anna",
        caller_number="+70001112233",
        context_name="sales",
        is_outbound=True,
        outbound_campaign_id="campaign-1",
        outbound_lead_id="lead-1",
        pre_call_results={},
        outbound_custom_vars={},
        external_platform=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_custom_vars_substitute_into_prompt_placeholders():
    engine = Engine.__new__(Engine)
    session = _substitution_session(
        outbound_custom_vars={"task": "confirm the appointment"}
    )

    result = engine._apply_prompt_template_substitution(
        "ROLE {caller_name}: {task}", session
    )

    assert result == "ROLE Anna: confirm the appointment"


def test_custom_vars_do_not_override_builtin_variables():
    engine = Engine.__new__(Engine)
    session = _substitution_session(
        outbound_custom_vars={"caller_name": "spoofed", "task": "call"}
    )

    result = engine._apply_prompt_template_substitution("{caller_name}: {task}", session)

    assert result == "Anna: call"


def test_builtin_placeholders_inside_custom_vars_values_resolve():
    engine = Engine.__new__(Engine)
    session = _substitution_session(
        outbound_custom_vars={"task": "call {caller_name} before {current_date}"}
    )

    result = engine._apply_prompt_template_substitution("{task}", session)

    assert result.startswith("call Anna before ")
    assert "{caller_name}" not in result
    assert "{current_date}" not in result


def test_long_custom_vars_values_substitute_untruncated():
    engine = Engine.__new__(Engine)
    long_prompt = "instruction " * 2000  # far beyond any former transport limit
    session = _substitution_session(outbound_custom_vars={"prompt": long_prompt})

    result = engine._apply_prompt_template_substitution("{prompt}", session)

    assert result == long_prompt


def test_unknown_placeholders_remain_untouched_with_custom_vars_merged():
    engine = Engine.__new__(Engine)
    session = _substitution_session(outbound_custom_vars={"task": "call"})

    result = engine._apply_prompt_template_substitution(
        "{task} keeps {unknown_var}", session
    )

    assert result == "call keeps {unknown_var}"


# ─── '## Lead Context' block: per-agent toggle, untruncated values ───────────


def test_lead_context_block_keeps_long_values_untruncated():
    engine = Engine.__new__(Engine)
    long_value = "y" * 1200

    result = engine._append_outbound_custom_vars_to_prompt("base", {"task": long_value})

    assert "## Lead Context" in result
    assert long_value in result


def test_lead_context_block_appended_by_default_and_when_enabled():
    engine = Engine.__new__(Engine)

    by_default = engine._append_outbound_custom_vars_to_prompt("base", {"task": "call"})
    enabled = engine._append_outbound_custom_vars_to_prompt(
        "base", {"task": "call"}, SimpleNamespace(lead_context_enabled=True)
    )
    legacy_row = engine._append_outbound_custom_vars_to_prompt(
        "base", {"task": "call"}, SimpleNamespace(lead_context_enabled=None)
    )

    for result in (by_default, enabled, legacy_row):
        assert "## Lead Context" in result


def test_lead_context_block_skipped_when_agent_disables_it():
    engine = Engine.__new__(Engine)

    result = engine._append_outbound_custom_vars_to_prompt(
        "base", {"task": "call"}, SimpleNamespace(lead_context_enabled=False)
    )

    assert result == "base"
