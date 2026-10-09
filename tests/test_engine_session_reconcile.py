"""The engine drops call sessions whose channel is gone from Asterisk, and call
setup stops as soon as the call is over.

Reproduces the ghost of 2026-09-29: an outbound call hung up 150 ms after
entering Stasis, cleanup ran, the still-running setup task bound an AudioSocket
leg and a late write put the session back; with no further Asterisk events the
session lived until the engine restarted.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.engine as engine_module
from src.core.models import CallSession
from src.core.session_store import SessionStore
from src.engine import Engine


def _engine(monkeypatch, *, channel_status=404, grace="0", interval="60", force_after="600") -> Engine:
    monkeypatch.setenv("AAVA_SESSION_ORPHAN_GRACE_SECONDS", grace)
    monkeypatch.setenv("AAVA_SESSION_RECONCILE_INTERVAL_SECONDS", interval)
    monkeypatch.setenv("AAVA_SESSION_ORPHAN_FORCE_SECONDS", force_after)
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.ari_client = MagicMock()
    engine.ari_client.running = True
    if channel_status == 200:
        engine.ari_client.send_command = AsyncMock(return_value={"id": "x", "state": "Up"})
    else:
        engine.ari_client.send_command = AsyncMock(return_value={"status": channel_status, "reason": "Channel not found"})
    engine.ari_client.hangup_channel = AsyncMock(return_value=True)
    engine.ari_client.destroy_bridge = AsyncMock(return_value=True)
    engine._cleanup_call = AsyncMock()
    engine._orphan_first_seen = {}
    engine._destroyed_channel_ts = {}
    return engine


async def _orphan(engine: Engine, call_id: str = "1790696444.14784", *, age: float = 120.0, cleaned: bool = False) -> CallSession:
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.created_at = time.time() - age
    session.cleanup_in_progress = cleaned
    session.status = "audiosocket_bound"
    session.conversation_state = "greeting"
    await engine.session_store.upsert_call(session)
    return session


@pytest.fixture(autouse=True)
def _clean_module_guards():
    engine_module._cleanup_in_progress.clear()
    engine_module._cleanup_completed_at.clear()
    yield
    engine_module._cleanup_in_progress.clear()
    engine_module._cleanup_completed_at.clear()


@pytest.mark.asyncio
async def test_a_resurrected_session_is_removed_without_a_second_cleanup(monkeypatch):
    engine = _engine(monkeypatch)
    session = await _orphan(engine, cleaned=True)   # cleanup already ran for this object

    removed = await engine._session_reconcile_once()

    assert removed == 1
    engine._cleanup_call.assert_not_awaited()          # no second round of history/emails
    assert await engine.session_store.get_by_call_id(session.call_id) is None
    assert engine.session_store.is_tombstoned(session.call_id)


@pytest.mark.asyncio
async def test_a_session_that_never_saw_cleanup_gets_the_normal_cleanup(monkeypatch):
    engine = _engine(monkeypatch)
    session = await _orphan(engine, cleaned=False)

    async def _real_cleanup(call_id, **kwargs):
        assert kwargs.get("ignore_ttl_guard") is True
        await engine.session_store.remove_call(call_id, tombstone=True)

    engine._cleanup_call = AsyncMock(side_effect=_real_cleanup)
    assert await engine._session_reconcile_once() == 1
    engine._cleanup_call.assert_awaited_once()
    assert await engine.session_store.get_by_call_id(session.call_id) is None


@pytest.mark.asyncio
async def test_a_session_that_survives_cleanup_is_removed_outright(monkeypatch):
    engine = _engine(monkeypatch)
    session = await _orphan(engine)
    assert await engine._session_reconcile_once() == 1      # the mocked cleanup leaves it in place
    assert await engine.session_store.get_by_call_id(session.call_id) is None


@pytest.mark.asyncio
async def test_orphan_must_be_gone_for_the_whole_grace_period(monkeypatch):
    engine = _engine(monkeypatch, grace="3600")
    session = await _orphan(engine, age=7200)
    assert await engine._session_reconcile_once() == 0       # first sighting only starts the clock
    assert session.call_id in engine._orphan_first_seen
    assert await engine.session_store.get_by_call_id(session.call_id) is not None


@pytest.mark.asyncio
async def test_fresh_sessions_live_channels_and_ari_errors_are_not_orphans(monkeypatch):
    engine = _engine(monkeypatch, grace="30")
    fresh = await _orphan(engine, "fresh", age=1)
    assert await engine._session_reconcile_once() == 0
    assert await engine.session_store.get_by_call_id(fresh.call_id) is not None

    engine = _engine(monkeypatch, channel_status=200)
    alive = await _orphan(engine, "alive")
    assert await engine._session_reconcile_once() == 0
    assert await engine.session_store.get_by_call_id(alive.call_id) is not None

    engine = _engine(monkeypatch, channel_status=500)
    flaky = await _orphan(engine, "flaky")
    assert await engine._session_reconcile_once() == 0
    assert "flaky" not in engine._orphan_first_seen


@pytest.mark.asyncio
async def test_no_reconciliation_while_ari_is_down(monkeypatch):
    engine = _engine(monkeypatch)
    engine.ari_client.running = False
    await _orphan(engine, cleaned=True)
    assert await engine._session_reconcile_once() == 0
    engine.ari_client.send_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_running_cleanup_is_left_alone_until_the_force_deadline(monkeypatch):
    engine = _engine(monkeypatch, force_after="600")
    session = await _orphan(engine)
    engine_module._cleanup_in_progress.add(session.call_id)
    engine._orphan_first_seen[session.call_id] = time.time() - 10
    assert await engine._session_reconcile_once() == 0
    assert await engine.session_store.get_by_call_id(session.call_id) is not None

    engine._orphan_first_seen[session.call_id] = time.time() - 601
    assert await engine._session_reconcile_once() == 1
    assert await engine.session_store.get_by_call_id(session.call_id) is None


def test_call_setup_aborts_once_the_call_is_over(monkeypatch):
    engine = _engine(monkeypatch)
    call_id = "1790696444.14784"
    assert engine._call_setup_aborted(call_id) is False

    engine._note_channel_destroyed(call_id)              # ChannelDestroyed arrived mid-setup
    assert engine._call_setup_aborted(call_id) is True

    other = "1790696444.14785"
    engine_module._cleanup_in_progress.add(other)         # cleanup running for another call
    assert engine._call_setup_aborted(other) is True
    engine_module._cleanup_in_progress.discard(other)
    engine_module._cleanup_completed_at[other] = time.time()
    assert engine._call_setup_aborted(other) is True


@pytest.mark.asyncio
async def test_operator_endpoint_reaps_a_ghost(monkeypatch):
    engine = _engine(monkeypatch)
    session = await _orphan(engine, cleaned=True)
    engine._is_request_authorized = lambda request: True
    request = MagicMock()
    request.match_info = {"call_id": session.call_id}
    request.query = {}

    response = await engine._session_cleanup_handler(request)

    assert response.status == 200
    assert await engine.session_store.get_by_call_id(session.call_id) is None
    missing = await engine._session_cleanup_handler(request)
    assert missing.status == 404
