"""The identity an outbound call is placed from.

A lead's Caller ID override used to be stored and shown but never dialed
with; every call went out as the global identity extension. It now
replaces that identity for the lead's own calls, on the channel and, on
FreePBX, as the extension the PBX believes placed the call.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.engine import Engine


def _originating_engine(*, pbx_type="freepbx", identity="6789"):
    engine = Engine.__new__(Engine)
    engine._outbound_extension_identity = identity
    engine._outbound_pbx_type = pbx_type
    engine._outbound_dial_context = "from-internal"
    engine._outbound_dial_prefix = ""
    engine._outbound_channel_tech = "local_only"
    engine._outbound_attempt_meta_by_attempt_id = {}
    engine._outbound_attempt_meta_by_channel_id = {}
    engine.providers = {}
    engine.config = SimpleNamespace(asterisk=SimpleNamespace(app_name="ai-voice-agent"))
    engine.transport_orchestrator = SimpleNamespace(get_context_config=lambda *a, **k: None)
    engine._outbound_agent_selector = lambda campaign, lead: ("sales", "ai_agent")
    engine._outbound_routing_channel_vars = lambda context_name, routing_method: {"AI_AGENT": context_name}
    engine._outbound_build_amd_opts = lambda options: ""
    engine.ari_client = SimpleNamespace(originate_channel=AsyncMock(return_value={"id": "chan-1"}))
    engine.outbound_store = SimpleNamespace(
        set_attempt_channel=AsyncMock(),
        finish_attempt=AsyncMock(),
        set_lead_state=AsyncMock(),
    )
    return engine


def _lead(**overrides):
    lead = {"id": "lead-1", "phone_number": "+15551230001", "name": "Иван"}
    lead.update(overrides)
    return lead


CAMPAIGN = {"id": "campaign-1", "voicemail_drop_enabled": 1, "consent_enabled": 0}


async def _originate(engine, lead):
    await engine._outbound_originate_attempt(CAMPAIGN, lead, "attempt-1")
    kwargs = engine.ari_client.originate_channel.call_args.kwargs
    return kwargs["caller_id"], kwargs["channel_vars"]


def test_the_lead_override_wins_and_a_blank_one_keeps_the_global_identity():
    engine = _originating_engine()
    assert engine._outbound_caller_identity(_lead(caller_id_override="101")) == ("101", "lead")
    assert engine._outbound_caller_identity(_lead(caller_id_override=" 101 ")) == ("101", "lead")
    assert engine._outbound_caller_identity(_lead()) == ("6789", "global")
    assert engine._outbound_caller_identity(_lead(caller_id_override="")) == ("6789", "global")
    assert engine._outbound_caller_identity(_lead(caller_id_override="   ")) == ("6789", "global")
    assert engine._outbound_caller_identity(_lead(caller_id_override=None)) == ("6789", "global")


@pytest.mark.asyncio
async def test_a_lead_with_an_override_is_dialed_as_that_extension_on_freepbx():
    engine = _originating_engine()

    caller_id, channel_vars = await _originate(engine, _lead(caller_id_override="101"))

    assert caller_id == "Asterisk AI <101>"
    for key in ("CALLERID(num)", "__CALLERID(num)", "AMPUSER", "__AMPUSER", "FROMEXTEN", "__FROMEXTEN"):
        assert channel_vars[key] == "101", key
    assert channel_vars["CALLERID(name)"] == "Asterisk AI"
    assert channel_vars["AAVA_LEAD_ID"] == "lead-1"


@pytest.mark.asyncio
async def test_a_lead_without_an_override_is_dialed_as_the_global_identity():
    engine = _originating_engine()

    caller_id, channel_vars = await _originate(engine, _lead())

    assert caller_id == "Asterisk AI <6789>"
    for key in ("CALLERID(num)", "__CALLERID(num)", "AMPUSER", "__AMPUSER", "FROMEXTEN", "__FROMEXTEN"):
        assert channel_vars[key] == "6789", key


@pytest.mark.asyncio
async def test_a_generic_pbx_gets_the_override_on_the_caller_id_only():
    engine = _originating_engine(pbx_type="generic")

    caller_id, channel_vars = await _originate(engine, _lead(caller_id_override="+74951234567"))

    assert caller_id == "Asterisk AI <+74951234567>"
    assert channel_vars["CALLERID(num)"] == "+74951234567"
    assert channel_vars["__CALLERID(num)"] == "+74951234567"
    assert "AMPUSER" not in channel_vars
    assert "FROMEXTEN" not in channel_vars
