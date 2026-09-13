"""A PlaybackFinished that cannot describe real audio must not reopen the mic.

Asterisk has been seen reporting a bridge playback finished within milliseconds
of accepting it. Trusting that cleared the TTS gating token while the prompt was
still audible, and in a modular pipeline the gated frames are dropped outright,
so the caller's own speech lost the slice cut out of it.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import CallSession, PlaybackRef
from src.core.playback_manager import MIN_PLAUSIBLE_PLAYBACK_SEC, PlaybackManager
from src.core.session_store import SessionStore

CALL_ID = "call-premature"
PLAYBACK_ID = "no-input-check_in:call-premature:1"


def _manager():
    ari = MagicMock()
    with patch("os.makedirs"):
        manager = PlaybackManager(SessionStore(), ari, "/tmp/test_media")
    manager.conversation_coordinator = MagicMock()
    manager.conversation_coordinator.on_tts_end = AsyncMock(return_value=True)
    manager.conversation_coordinator.update_conversation_state = AsyncMock()
    manager._cleanup_audio_file = AsyncMock()
    return manager


async def _register(manager, *, duration_sec, started_ago_sec):
    session = CallSession(call_id=CALL_ID, caller_channel_id=CALL_ID)
    await manager.session_store.upsert_call(session)
    await manager.session_store.add_playback(
        PlaybackRef(
            playback_id=PLAYBACK_ID,
            call_id=CALL_ID,
            channel_id=CALL_ID,
            bridge_id="bridge-1",
            media_uri="sound:ai-generated/x",
            audio_file="/tmp/x.ulaw",
            expected_duration_sec=duration_sec,
        )
    )
    manager._playback_started_at[PLAYBACK_ID] = time.monotonic() - started_ago_sec


@pytest.mark.asyncio
async def test_an_instant_finish_is_ignored():
    """The reported case: 2.88 s of audio, event 3 ms after the start."""
    manager = _manager()
    await _register(manager, duration_sec=2.88, started_ago_sec=0.003)

    handled = await manager.on_playback_finished(PLAYBACK_ID)

    assert handled is False
    manager.conversation_coordinator.on_tts_end.assert_not_awaited()
    # The playback is still tracked, so the scheduled fallback can clear it.
    assert await manager.session_store.get_playback(PLAYBACK_ID) is not None


@pytest.mark.asyncio
async def test_a_finish_after_the_audio_played_is_honoured():
    manager = _manager()
    await _register(manager, duration_sec=2.88, started_ago_sec=2.9)

    handled = await manager.on_playback_finished(PLAYBACK_ID)

    assert handled is True
    manager.conversation_coordinator.on_tts_end.assert_awaited_once()
    assert await manager.session_store.get_playback(PLAYBACK_ID) is None


@pytest.mark.asyncio
async def test_the_floor_is_the_only_thing_checked():
    """Just past the floor counts as real, even well before the audio ends."""
    manager = _manager()
    await _register(
        manager, duration_sec=2.88, started_ago_sec=MIN_PLAUSIBLE_PLAYBACK_SEC + 0.01
    )

    assert await manager.on_playback_finished(PLAYBACK_ID) is True


@pytest.mark.asyncio
async def test_short_audio_is_not_guarded():
    """A clip shorter than twice the floor can legitimately finish inside it."""
    manager = _manager()
    await _register(manager, duration_sec=0.2, started_ago_sec=0.01)

    assert await manager.on_playback_finished(PLAYBACK_ID) is True


@pytest.mark.asyncio
async def test_a_playback_with_no_recorded_start_is_not_guarded():
    manager = _manager()
    await _register(manager, duration_sec=2.88, started_ago_sec=0.003)
    manager._playback_started_at.pop(PLAYBACK_ID)

    assert await manager.on_playback_finished(PLAYBACK_ID) is True


@pytest.mark.asyncio
async def test_an_unknown_playback_still_reports_false():
    manager = _manager()

    assert await manager.on_playback_finished("never-seen") is False


@pytest.mark.asyncio
async def test_the_start_time_is_forgotten_once_the_finish_is_accepted():
    manager = _manager()
    await _register(manager, duration_sec=2.88, started_ago_sec=2.9)

    await manager.on_playback_finished(PLAYBACK_ID)

    assert PLAYBACK_ID not in manager._playback_started_at


def test_duration_is_computed_from_the_telephony_rate():
    with patch("os.makedirs"):
        manager = PlaybackManager(SessionStore(), MagicMock(), "/tmp/test_media")

    assert manager._audio_duration_sec(8000) == pytest.approx(1.0)
    assert manager._audio_duration_sec(0) == 0.0
    assert manager._audio_duration_sec(-5) == 0.0
