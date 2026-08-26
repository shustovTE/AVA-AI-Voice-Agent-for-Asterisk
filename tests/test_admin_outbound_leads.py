import importlib.util

import pytest

# admin_ui backend imports fastapi at module load. Skip the whole module on
# environments that don't have it (CI's engine-only jobs run without admin_ui
# deps), matching tests/test_admin_outbound_recordings.py.
if importlib.util.find_spec("fastapi") is None:
    pytest.skip("fastapi not installed; admin_ui outbound tests skipped", allow_module_level=True)

from fastapi import HTTPException

from admin_ui.backend.api import outbound


async def _campaign_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL_HISTORY_ENABLED", "true")
    from src.core.outbound_store import OutboundStore

    store = OutboundStore(db_path=str(tmp_path / "call_history.db"))
    campaign = await store.create_campaign(
        {
            "name": "Test Campaign",
            "timezone": "UTC",
            "daily_window_start_local": "09:00",
            "daily_window_end_local": "17:00",
            "max_concurrent": 1,
            "min_interval_seconds_between_calls": 0,
            "default_context": "demo",
            "voicemail_drop_mode": "upload",
            "voicemail_drop_media_uri": "sound:ai-generated/test-vm",
        }
    )
    monkeypatch.setattr(outbound, "_get_outbound_store", lambda: store)
    monkeypatch.setattr(
        outbound, "_load_known_agent_selectors", lambda: (["demo", "demo2"], [])
    )
    return store, campaign["id"]


@pytest.mark.asyncio
async def test_add_manual_lead_duplicate_returns_existing_lead_id(tmp_path, monkeypatch):
    store, campaign_id = await _campaign_store(tmp_path, monkeypatch)

    created = await outbound.add_manual_lead(
        campaign_id,
        outbound.ManualLeadCreateRequest(
            phone_number="+15551230001", custom_vars={"task": "first"}
        ),
    )
    assert created.accepted == 1
    assert created.duplicates == 0
    assert created.duplicate_lead_id is None

    # Same number in a different formatting still resolves the existing lead.
    duplicate = await outbound.add_manual_lead(
        campaign_id,
        outbound.ManualLeadCreateRequest(
            phone_number="+1 (555) 123-0001", custom_vars={"task": "second"}
        ),
    )
    assert duplicate.accepted == 0
    assert duplicate.duplicates == 1
    assert duplicate.duplicate_lead_id
    assert (
        await store.get_lead_id_by_phone(campaign_id, "+15551230001")
        == duplicate.duplicate_lead_id
    )


@pytest.mark.asyncio
async def test_patch_lead_updates_fields_and_maps_errors(tmp_path, monkeypatch):
    store, campaign_id = await _campaign_store(tmp_path, monkeypatch)
    created = await outbound.add_manual_lead(
        campaign_id,
        outbound.ManualLeadCreateRequest(
            phone_number="+15551230002", custom_vars={"task": "old"}
        ),
    )
    assert created.accepted == 1
    lead_id = await store.get_lead_id_by_phone(campaign_id, "+15551230002")

    long_prompt = "instruction " * 2000
    patched = await outbound.patch_lead(
        lead_id,
        outbound.LeadPatchRequest(
            name="Anna",
            agent="demo2",
            caller_id="101",
            custom_vars={"task": "new", "prompt": long_prompt},
        ),
    )
    assert patched["name"] == "Anna"
    assert patched["context_override"] == "demo2"
    assert patched["agent_routing_method"] == "ai_agent"
    assert patched["caller_id_override"] == "101"
    assert patched["custom_vars"] == {"task": "new", "prompt": long_prompt}

    # Explicit null clears the override; unset fields stay untouched.
    cleared = await outbound.patch_lead(
        lead_id, outbound.LeadPatchRequest(caller_id=None)
    )
    assert cleared["caller_id_override"] is None
    assert cleared["name"] == "Anna"

    with pytest.raises(HTTPException) as exc:
        await outbound.patch_lead(lead_id, outbound.LeadPatchRequest(agent="ghost"))
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await outbound.patch_lead("missing-lead", outbound.LeadPatchRequest(name="x"))
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        await outbound.patch_lead(lead_id, outbound.LeadPatchRequest())
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_patch_lead_conflicts_while_lead_is_active(tmp_path, monkeypatch):
    store, campaign_id = await _campaign_store(tmp_path, monkeypatch)
    await outbound.add_manual_lead(
        campaign_id,
        outbound.ManualLeadCreateRequest(phone_number="+15551230003"),
    )
    lead_id = await store.get_lead_id_by_phone(campaign_id, "+15551230003")

    leased = await store.lease_pending_leads(campaign_id, limit=1, lease_seconds=60)
    assert leased and leased[0]["id"] == lead_id

    with pytest.raises(HTTPException) as exc:
        await outbound.patch_lead(lead_id, outbound.LeadPatchRequest(name="Nope"))
    assert exc.value.status_code == 409

    # Finished leads are editable again (PATCH + recycle is the redial flow).
    await store.set_lead_state(lead_id, state="completed", last_outcome="answered_human")
    patched = await outbound.patch_lead(
        lead_id, outbound.LeadPatchRequest(custom_vars={"task": "next call"})
    )
    assert patched["custom_vars"] == {"task": "next call"}
    assert await store.recycle_lead(lead_id, mode="redial") is True
