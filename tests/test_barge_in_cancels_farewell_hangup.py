"""A barge-in during a farewell cancels the hangup the farewell had armed.

With "Hang up on assistant farewell" on, a pipeline reply ending in a farewell
marker started a terminal hangup that waited for the audio to drain and then
hung up. A caller who interrupted that farewell stopped the stream, which
looked like a drained transport, and was hung up on mid-sentence. The hangup
was a guess about the conversation, so a barge-in now cancels it (and the
assistant-farewell marker's fallback); an explicit hangup_call is left alone.
"""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine


def _config() -> AppConfig:
    return AppConfig(
        **{
            "default_provider": "local",
            "providers": {"local": {"enabled": True}},
            "asterisk": {
                "host": "127.0.0.1",
                "port": 8088,
                "username": "u",
                "password": "p",
                "app_name": "ai-voice-agent",
            },
            "llm": {"initial_greeting": "", "prompt": "You are helpful", "model": "gpt-4o"},
            "pipelines": {"serial": {}},
            "active_pipeline": "serial",
            "audio_transport": "audiosocket",
            "downstream_mode": "stream",
        }
    )


class _PlaybackStub:
    def __init__(self):
        self.active_streams = {}
        self.stopped = 0

    def is_stream_active(self, call_id, stream_id=None):
        return False

    def get_playback_position_ms(self, call_id):
        return 0

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.stopped += 1


async def _engine(monkeypatch):
    engine = Engine(_config())
    call_id = f"call-farewell-{uuid4().hex[:8]}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.media_rx_confirmed = True
    await engine.session_store.upsert_call(session)
    monkeypatch.setattr(engine, "streaming_playback_manager", _PlaybackStub())
    pending = {"bytes": 3200}
    engine._call_audio_drain_snapshot = AsyncMock(
        side_effect=lambda cid: {"pending_stream_bytes": pending["bytes"], "last_real_emit_ts": None}
    )
    engine._terminal_transport_quiet_sec = lambda: 0.05
    engine.ari_client.hangup_channel = AsyncMock(return_value=True)
    return engine, session, pending


async def _barge_in(engine, call_id):
    await engine._apply_barge_in_action(call_id, source="silero_vad", reason="pipeline_tts_overlap")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, "agent_hangup"])
async def test_a_barge_in_during_the_farewell_drain_keeps_the_call_alive(monkeypatch, outcome):
    engine, session, pending = await _engine(monkeypatch)
    terminate = asyncio.create_task(
        engine._terminate_call_after_audio(
            session.call_id, reason="pipeline_farewell_without_tool", call_outcome=outcome
        )
    )
    await asyncio.sleep(0.1)  # the farewell is still draining
    assert session.call_id in engine._terminal_hangup_started

    await _barge_in(engine, session.call_id)
    pending["bytes"] = 0  # the stopped stream looks drained at once

    assert await asyncio.wait_for(terminate, timeout=2) is False
    engine.ari_client.hangup_channel.assert_not_awaited()
    assert session.call_id not in engine._terminal_hangup_started
    assert session.call_id not in engine._terminal_hangup_cancelled
    assert session.call_outcome != "agent_hangup"
    # The next farewell ends the call as usual.
    assert await engine._terminate_call_after_audio(
        session.call_id, reason="pipeline_farewell_without_tool", call_outcome="agent_hangup"
    ) is True
    engine.ari_client.hangup_channel.assert_awaited_once_with(session.call_id)


@pytest.mark.asyncio
async def test_a_barge_in_leaves_an_explicit_hangup_call_alone(monkeypatch):
    engine, session, pending = await _engine(monkeypatch)
    terminate = asyncio.create_task(
        engine._terminate_call_after_audio(
            session.call_id, reason="pipeline_hangup_call", call_outcome="agent_hangup"
        )
    )
    await asyncio.sleep(0.1)

    await _barge_in(engine, session.call_id)
    pending["bytes"] = 0

    assert await asyncio.wait_for(terminate, timeout=2) is True
    engine.ari_client.hangup_channel.assert_awaited_once_with(session.call_id)
    assert session.call_outcome == "agent_hangup"


@pytest.mark.asyncio
async def test_a_barge_in_disarms_the_assistant_farewell_marker(monkeypatch):
    """The marker flow arms cleanup_after_tts and a fallback timer; both are dropped."""
    engine, session, pending = await _engine(monkeypatch)
    session.cleanup_after_tts = True
    engine._schedule_terminal_fallback(
        session.call_id, reason="assistant_farewell_marker", timeout_sec=15.0, call_outcome="agent_hangup"
    )
    fallback = engine._terminal_fallback_tasks[session.call_id]
    assert engine._terminal_fallback_reasons[session.call_id] == "assistant_farewell_marker"

    await _barge_in(engine, session.call_id)
    await asyncio.sleep(0)

    assert session.cleanup_after_tts is False
    assert fallback.cancelled() or fallback.done()
    assert session.call_id not in engine._terminal_fallback_tasks
    assert session.call_id not in engine._terminal_fallback_reasons
    assert session.call_outcome == ""
    # The generic audio-done hangup that the marker would have triggered no longer counts as a guess.
    assert engine._terminal_hangup_is_heuristic(session.call_id, "cleanup_after_tts") is False


@pytest.mark.asyncio
async def test_a_barge_in_leaves_a_tool_fallback_timer_alone(monkeypatch):
    engine, session, pending = await _engine(monkeypatch)
    session.cleanup_after_tts = True
    engine._schedule_terminal_fallback(
        session.call_id, reason="local:hangup_call", timeout_sec=15.0, call_outcome="agent_hangup"
    )
    fallback = engine._terminal_fallback_tasks[session.call_id]

    await _barge_in(engine, session.call_id)
    await asyncio.sleep(0)

    assert session.cleanup_after_tts is True
    assert not fallback.done()
    fallback.cancel()


def test_which_hangups_are_guesses():
    engine = Engine.__new__(Engine)
    engine._terminal_fallback_reasons = {"c1": "assistant_farewell_marker"}
    assert engine._terminal_hangup_is_heuristic("c1", "pipeline_farewell_without_tool") is True
    assert engine._terminal_hangup_is_heuristic("c1", "assistant_farewell_marker:fallback_timeout") is True
    assert engine._terminal_hangup_is_heuristic("c1", "cleanup_after_tts") is True
    assert engine._terminal_hangup_is_heuristic("c2", "cleanup_after_tts") is False
    assert engine._terminal_hangup_is_heuristic("c1", "pipeline_hangup_call") is False
    assert engine._terminal_hangup_is_heuristic("c1", None) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reason, outcome", [
    ("pipeline_hangup_call", None),
    ("pipeline_followup_hangup_call", None),
    ("local_asterisk_farewell", None),
    ("local_farewell_timeout", None),
    ("elevenlabs:hangup_call:fallback_timeout", None),
    ("pipeline_farewell_without_tool", "agent_hangup"),
    ("cleanup_after_tts", "agent_hangup"),
    ("assistant_farewell_marker:fallback_timeout", "agent_hangup"),
])
async def test_farewell_hangup_reaches_post_call_webhook_as_agent(monkeypatch, reason, outcome):
    from src.tools.http.generic_webhook import GenericWebhookTool, WebhookConfig

    engine, session, pending = await _engine(monkeypatch)
    session.cleanup_after_tts = True
    pending["bytes"] = 0
    webhook = GenericWebhookTool(WebhookConfig(
        name="end_of_call",
        payload_template='{"call_outcome": "{call_outcome}"}',
    ))
    engine._post_call_tools_for_context = lambda *args: [webhook]
    engine._run_post_call_tools = AsyncMock()
    engine._persist_call_history = AsyncMock()

    # Several production paths omit call_outcome. The ARI disconnect immediately
    # starts cleanup, which must receive the outcome before the TTS flag clears.
    outcomes_at_hangup = []

    async def hangup(channel_id):
        outcomes_at_hangup.append(session.call_outcome)
        await engine._cleanup_call(channel_id)
        return True

    engine.ari_client.hangup_channel.side_effect = hangup
    assert await engine._terminate_call_after_audio(
        session.call_id, reason=reason, call_outcome=outcome,
    )

    engine._run_post_call_tools.assert_awaited_once()
    context = engine._run_post_call_tools.await_args.args[2]
    assert json.loads(webhook._build_payload(context)) == {"call_outcome": "agent_hangup"}
    assert outcomes_at_hangup[0] == "agent_hangup"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["no_input_timeout", "max_duration"])
async def test_terminal_hangup_preserves_existing_policy_outcome(monkeypatch, outcome):
    engine, session, pending = await _engine(monkeypatch)
    session.call_outcome = outcome
    pending["bytes"] = 0

    assert await engine._terminate_call_after_audio(session.call_id, reason="test")
    assert session.call_outcome == outcome


@pytest.mark.asyncio
async def test_caller_disconnect_after_cancelled_farewell_stays_caller_hangup(monkeypatch):
    engine, session, pending = await _engine(monkeypatch)
    session.cleanup_after_tts = True
    engine._schedule_terminal_fallback(
        session.call_id, reason="assistant_farewell_marker", call_outcome="agent_hangup",
    )
    await _barge_in(engine, session.call_id)
    engine._execute_post_call_tools = AsyncMock()
    engine._persist_call_history = AsyncMock()

    await engine._cleanup_call(session.call_id)

    engine._execute_post_call_tools.assert_awaited_once()
    assert engine._execute_post_call_tools.await_args.kwargs["call_outcome"] == "caller_hangup"
