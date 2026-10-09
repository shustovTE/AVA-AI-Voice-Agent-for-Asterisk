"""A call session removed by cleanup must stay removed.

Late writers (a pipeline task finishing its request, a coordinator timer, an
AudioSocket bind that raced the hangup) hold the removed CallSession object and
call upsert_call with it; without a tombstone that write resurrected a session
whose channel was gone, and nothing ever removed it again.
"""

from __future__ import annotations

import time

import pytest

from src.core.models import CallSession
from src.core.session_store import SessionStore


def _session(call_id: str = "1790696444.14784") -> CallSession:
    return CallSession(call_id=call_id, caller_channel_id=call_id)


@pytest.mark.asyncio
async def test_cleanup_removal_tombstones_the_call_and_refuses_a_late_write():
    store = SessionStore()
    session = _session()
    assert await store.upsert_call(session) is True

    removed = await store.remove_call(session.call_id, tombstone=True)
    assert removed is session
    assert store.is_tombstoned(session.call_id)

    # The stale object comes back from a late writer: refused, store stays empty.
    session.status = "audiosocket_bound"
    assert await store.upsert_call(session) is False
    assert await store.get_by_call_id(session.call_id) is None
    assert await store.get_by_channel_id(session.caller_channel_id) is None
    assert (await store.get_session_stats())["active_calls"] == 0


@pytest.mark.asyncio
async def test_a_live_session_is_still_updated_after_an_old_tombstone():
    """The tombstone only blocks re-creation; updates of a session that exists are fine."""
    store = SessionStore()
    session = _session()
    await store.upsert_call(session)
    await store.remove_call(session.call_id, tombstone=True)
    await store.upsert_call(session, allow_resurrect=True)   # deliberate rebuild
    session.status = "listening"
    assert await store.upsert_call(session) is True
    assert (await store.get_by_call_id(session.call_id)).status == "listening"


@pytest.mark.asyncio
async def test_plain_removal_does_not_tombstone_and_forget_lifts_it():
    store = SessionStore()
    session = _session()
    await store.upsert_call(session)
    await store.remove_call(session.call_id)          # e.g. a reconstructed VICIdial session
    assert not store.is_tombstoned(session.call_id)
    assert await store.upsert_call(session) is True

    await store.remove_call(session.call_id, tombstone=True)
    await store.forget_tombstone(session.call_id)
    assert await store.upsert_call(session) is True


@pytest.mark.asyncio
async def test_tombstones_expire_and_stay_bounded():
    store = SessionStore()
    store._tombstones["old"] = time.time() - store.TOMBSTONE_TTL_SECONDS - 1
    await store.remove_call("fresh", tombstone=True)
    assert "old" not in store._tombstones and "fresh" in store._tombstones

    for i in range(store.TOMBSTONE_MAX_ENTRIES + 5):
        store._tombstones[f"c{i}"] = time.time() + i * 1e-6
    await store.remove_call("last", tombstone=True)
    assert len(store._tombstones) <= store.TOMBSTONE_MAX_ENTRIES


@pytest.mark.asyncio
async def test_session_stats_expose_age_and_cleanup_marker():
    store = SessionStore()
    session = _session()
    session.created_at = time.time() - 120
    session.cleanup_in_progress = True
    session.is_outbound = True
    await store.upsert_call(session)
    stats = await store.get_session_stats()
    row = stats["sessions"][0]
    assert row["caller_channel_id"] == session.call_id
    assert row["cleanup_in_progress"] is True and row["is_outbound"] is True
    assert 119 <= row["age_seconds"] <= 125
