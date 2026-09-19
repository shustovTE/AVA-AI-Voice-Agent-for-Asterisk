import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.models import CallSession
from src.core.no_input_watchdog import NoInputPolicy, NoInputWatchdog
from src.core.conversation_coordinator import ConversationCoordinator
from src.core.session_store import SessionStore
from src.config import NoInputConfig
from src.engine import Engine


async def _wait_until(predicate, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.005)


def test_policy_coerces_raw_overrides_without_truthy_string_or_numeric_surprises():
    policy = NoInputPolicy.from_mapping(
        {
            "enabled": "false",
            "inbound_enabled": "0",
            "outbound_enabled": "yes",
            "initial_timeout_sec": float("nan"),
            "grace_timeout_sec": 5000,
            "max_check_ins": 1.5,
            "check_in_message": "   ",
            "final_message": "",
        }
    )

    assert policy.enabled is False
    assert policy.inbound_enabled is False
    assert policy.outbound_enabled is True
    assert policy.initial_timeout_sec == 30.0
    assert policy.grace_timeout_sec == 15.0
    assert policy.max_check_ins == 1
    assert policy.check_in_message == "Are you still there?"
    assert policy.final_message == "I still can't hear you, so I'll end the call now. Goodbye."


def test_global_config_rejects_blank_announcement_messages():
    with pytest.raises(ValueError):
        NoInputConfig(final_message="   ")
    with pytest.raises(ValueError):
        NoInputConfig(check_in_message="")


@pytest.mark.asyncio
async def test_watchdog_checks_in_then_says_final_message_and_hangs_up():
    announcements = []
    hangups = []

    async def announce(call_id, text, kind):
        announcements.append((call_id, text, kind))
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(
        initial_timeout_sec=0.04,
        grace_timeout_sec=0.03,
        max_check_ins=1,
        check_in_message="Still there?",
        final_message="Goodbye.",
    )
    await watchdog.register("call-1", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-1")
        await _wait_until(lambda: hangups == ["call-1"])
        assert announcements == [
            ("call-1", "Still there?", "check_in"),
            ("call-1", "Goodbye.", "final"),
        ]
        assert watchdog.snapshot("call-1")["phase"] == "hangup"
    finally:
        await watchdog.stop("call-1")


@pytest.mark.asyncio
async def test_final_announcement_exception_still_attempts_hangup():
    hangup = AsyncMock()

    async def announce(_call_id, _text, kind):
        if kind == "final":
            raise RuntimeError("provider unavailable")
        return True

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(
        initial_timeout_sec=0.03,
        grace_timeout_sec=0.02,
        max_check_ins=0,
    )
    await watchdog.register("call-final-failure", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-final-failure")
        await _wait_until(lambda: hangup.await_count == 1)
        assert watchdog.snapshot("call-final-failure")["phase"] == "hangup"
    finally:
        await watchdog.stop("call-final-failure")


@pytest.mark.asyncio
async def test_check_in_announcement_exception_keeps_watchdog_running():
    kinds = []
    hangup = AsyncMock()

    async def announce(_call_id, _text, kind):
        kinds.append(kind)
        if kind == "check_in":
            raise RuntimeError("provider unavailable")
        return True

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(
        initial_timeout_sec=0.03,
        grace_timeout_sec=0.02,
        max_check_ins=1,
    )
    await watchdog.register("call-check-in-failure", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-check-in-failure")
        await _wait_until(lambda: hangup.await_count == 1)
        assert kinds == ["check_in", "final"]
    finally:
        await watchdog.stop("call-check-in-failure")


@pytest.mark.asyncio
async def test_unexpected_watchdog_failure_removes_active_state():
    async def should_pause(_call_id):
        raise RuntimeError("session store unavailable")

    watchdog = NoInputWatchdog(AsyncMock(return_value=True), AsyncMock(), should_pause=should_pause)
    policy = NoInputPolicy(initial_timeout_sec=0.02, grace_timeout_sec=0.02, max_check_ins=1)
    await watchdog.register("call-run-failure", policy, is_outbound=False)
    await watchdog.mark_ready("call-run-failure")

    await _wait_until(lambda: not watchdog.has_call("call-run-failure"))


@pytest.mark.asyncio
async def test_caller_activity_resets_the_initial_window():
    announcements = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    watchdog = NoInputWatchdog(announce, AsyncMock())
    policy = NoInputPolicy(initial_timeout_sec=0.06, grace_timeout_sec=0.03, max_check_ins=1)
    await watchdog.register("call-2", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-2")
        await asyncio.sleep(0.04)
        await watchdog.note_activity("call-2", "test:transcript")
        await asyncio.sleep(0.04)
        assert announcements == []
        await _wait_until(lambda: announcements == ["check_in"])
    finally:
        await watchdog.stop("call-2")


@pytest.mark.asyncio
async def test_sustained_caller_speech_and_agent_output_pause_the_clock():
    announcements = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    watchdog = NoInputWatchdog(announce, AsyncMock())
    policy = NoInputPolicy(initial_timeout_sec=0.04, grace_timeout_sec=0.03, max_check_ins=1)
    await watchdog.register("call-3", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-3")
        await watchdog.note_input_state("call-3", True, "test:audio")
        await asyncio.sleep(0.08)
        assert announcements == []
        await watchdog.note_input_state("call-3", False, "test:audio")
        await watchdog.note_agent_output_start("call-3")
        await asyncio.sleep(0.08)
        assert announcements == []
        await watchdog.note_agent_output_end("call-3")
        await _wait_until(lambda: announcements == ["check_in"])
    finally:
        await watchdog.stop("call-3")


@pytest.mark.asyncio
async def test_unmatched_input_end_does_not_extend_check_in_grace_deadline():
    watchdog = NoInputWatchdog(AsyncMock(return_value=True), AsyncMock(), clock=lambda: 1000.0)
    policy = NoInputPolicy(initial_timeout_sec=30.0, grace_timeout_sec=15.0, max_check_ins=1)
    await watchdog.register("call-unmatched-end", policy, is_outbound=False)
    try:
        state = watchdog._states["call-unmatched-end"]
        state.ready = True
        state.input_active = False
        state.phase = "grace"
        state.check_ins = 1
        state.deadline = 1015.0

        await watchdog.note_input_state(
            "call-unmatched-end",
            False,
            "asterisk:talk_detect",
        )

        snapshot = watchdog.snapshot("call-unmatched-end")
        assert snapshot["phase"] == "grace"
        assert snapshot["check_ins"] == 1
        assert snapshot["deadline"] == 1015.0
    finally:
        await watchdog.stop("call-unmatched-end")


@pytest.mark.asyncio
async def test_hosted_silence_output_pauses_without_resetting_deadline():
    announcements = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    watchdog = NoInputWatchdog(announce, AsyncMock())
    policy = NoInputPolicy(initial_timeout_sec=0.08, grace_timeout_sec=0.03, max_check_ins=1)
    await watchdog.register("call-hosted-silence", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-hosted-silence")
        await asyncio.sleep(0.05)
        await watchdog.note_agent_output_start("call-hosted-silence")
        await asyncio.sleep(0.05)
        assert announcements == []
        await watchdog.note_agent_output_end("call-hosted-silence", reset_timer=False)
        # Only the ~30ms remaining before hosted output should be restored.
        await _wait_until(lambda: announcements == ["check_in"], timeout=0.07)
    finally:
        await watchdog.stop("call-hosted-silence")


@pytest.mark.asyncio
async def test_self_announcement_drain_completion_preserves_grace_state():
    watchdog = NoInputWatchdog(AsyncMock(return_value=True), AsyncMock())
    policy = NoInputPolicy(initial_timeout_sec=30.0, grace_timeout_sec=15.0, max_check_ins=1)
    await watchdog.register("call-self-announcement", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-self-announcement")
        state = watchdog._states["call-self-announcement"]
        state.output_active = True
        state.self_announcement = False
        state.phase = "grace"
        state.check_ins = 1
        state.deadline = 1234.5

        await watchdog.note_agent_output_end(
            "call-self-announcement",
            reset_timer=True,
            preserve_policy_state=True,
        )

        snapshot = watchdog.snapshot("call-self-announcement")
        assert snapshot["output_active"] is False
        assert snapshot["phase"] == "grace"
        assert snapshot["check_ins"] == 1
        assert snapshot["deadline"] == 1234.5
    finally:
        await watchdog.stop("call-self-announcement")


@pytest.mark.asyncio
async def test_raw_audio_detector_ignores_audio_during_native_provider_output():
    engine = Engine.__new__(Engine)
    engine.no_input_watchdog = SimpleNamespace(
        has_call=lambda _call_id: True,
        note_input_state=AsyncMock(),
    )
    engine._agent_output_active_calls = {"call-native-output"}
    session = CallSession(
        call_id="call-native-output",
        caller_channel_id="channel-native-output",
    )
    session.tts_playing = False

    await engine._observe_no_input_audio(
        session,
        b"\xff\x7f" * 160,
        16000,
        source="audiosocket",
    )

    engine.no_input_watchdog.note_input_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_talk_detect_echo_tail_does_not_pause_no_input_watchdog():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.config = SimpleNamespace(
        barge_in=SimpleNamespace(enabled=True, post_tts_end_protection_ms=600)
    )
    engine._no_input_note_input_state = AsyncMock()
    session = CallSession(
        call_id="call-post-tts-echo",
        caller_channel_id="channel-post-tts-echo",
    )
    session.audio_capture_enabled = True
    session.tts_playing = False
    session.tts_ended_ts = time.time()
    await engine.session_store.upsert_call(session)

    await engine._handle_channel_talking_started(
        {"channel": {"id": session.caller_channel_id}}
    )

    engine._no_input_note_input_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_talk_detect_after_post_tts_guard_pauses_no_input_watchdog():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.config = SimpleNamespace(
        barge_in=SimpleNamespace(enabled=True, post_tts_end_protection_ms=600)
    )
    engine._no_input_note_input_state = AsyncMock()
    session = CallSession(
        call_id="call-real-talking",
        caller_channel_id="channel-real-talking",
    )
    session.audio_capture_enabled = True
    session.tts_playing = False
    session.tts_ended_ts = time.time() - 1.0
    await engine.session_store.upsert_call(session)

    await engine._handle_channel_talking_started(
        {"channel": {"id": session.caller_channel_id}}
    )

    engine._no_input_note_input_state.assert_awaited_once_with(
        session.call_id,
        True,
        "asterisk:talk_detect",
    )


@pytest.mark.asyncio
async def test_talk_detect_rechecks_gating_after_async_cleanup_race():
    """An echo event queued during TTS must not interrupt after TTS ends."""
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.config = SimpleNamespace(
        barge_in=SimpleNamespace(
            enabled=True,
            post_tts_end_protection_ms=600,
            talk_detect_initial_protection_ms=0,
            cooldown_ms=0,
        )
    )
    engine._no_input_note_input_state = AsyncMock()
    engine._apply_barge_in_action = AsyncMock()
    engine._save_session = AsyncMock()
    session = CallSession(
        call_id="call-stale-talk-detect",
        caller_channel_id="channel-stale-talk-detect",
    )
    session.audio_capture_enabled = False
    session.tts_playing = True
    session.tts_started_ts = time.time() - 2.0
    await engine.session_store.upsert_call(session)

    async def finish_playback_during_handler(*_args, **_kwargs):
        session.tts_playing = False
        session.audio_capture_enabled = True
        session.tts_ended_ts = time.time()

    engine._no_input_note_activity = AsyncMock(
        side_effect=finish_playback_during_handler
    )

    await engine._handle_channel_talking_started(
        {"channel": {"id": session.caller_channel_id}}
    )

    engine._apply_barge_in_action.assert_not_awaited()
    engine._no_input_note_input_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_input_provider_output_drains_without_resetting_policy_state():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine._call_bg_tasks = {}
    engine._provider_output_operations = {}
    engine._provider_output_drain_tasks = {}
    engine._agent_output_active_calls = set()
    engine.config = SimpleNamespace(audio_transport="audiosocket")
    engine.no_input_watchdog = SimpleNamespace(
        note_agent_output_start=AsyncMock(),
        note_agent_output_end=AsyncMock(),
    )
    engine._save_session = AsyncMock()
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()

    async def wait_for_drain(_call_id, **_kwargs):
        drain_started.set()
        await release_drain.wait()
        return True

    engine._wait_for_call_audio_drain = wait_for_drain
    session = CallSession(
        call_id="call-provider-tail",
        caller_channel_id="channel-provider-tail",
    )
    await engine.session_store.upsert_call(session)
    engine._provider_output_operations[session.call_id] = {
        "output_id": "no-input-check-in",
        "purpose": "no_input_check_in",
        "audio_started": asyncio.Event(),
        "generation_done": asyncio.Event(),
    }

    await engine._note_provider_output_start(session.call_id)
    await engine._note_provider_output_end(session.call_id, session)
    await asyncio.wait_for(drain_started.wait(), timeout=0.2)

    assert session.call_id in engine._agent_output_active_calls
    engine.no_input_watchdog.note_agent_output_end.assert_not_awaited()

    release_drain.set()
    await _wait_until(lambda: session.call_id not in engine._agent_output_active_calls)
    engine.no_input_watchdog.note_agent_output_end.assert_awaited_once_with(
        session.call_id,
        reset_timer=True,
        preserve_policy_state=True,
    )

    await engine._note_provider_output_end(session.call_id, session)
    assert engine.no_input_watchdog.note_agent_output_end.await_count == 2
    assert engine.no_input_watchdog.note_agent_output_end.await_args.kwargs == {
        "reset_timer": True,
        "preserve_policy_state": True,
    }


@pytest.mark.asyncio
async def test_openai_greeting_gating_is_released_only_after_transport_drain():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.conversation_coordinator = ConversationCoordinator(engine.session_store)
    engine._call_bg_tasks = {}
    engine._provider_output_operations = {}
    engine._provider_output_drain_tasks = {}
    engine._agent_output_active_calls = set()
    greeting_provider = SimpleNamespace(
        release_greeting_transport_guard=AsyncMock(),
    )
    engine._call_providers = {"call-openai-greeting-tail": greeting_provider}
    engine.config = SimpleNamespace(audio_transport="audiosocket")
    engine.no_input_watchdog = SimpleNamespace(
        note_agent_output_start=AsyncMock(),
        note_agent_output_end=AsyncMock(),
    )
    engine._save_session = AsyncMock()
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()

    async def wait_for_drain(_call_id, **_kwargs):
        drain_started.set()
        await release_drain.wait()
        return True

    engine._wait_for_call_audio_drain = wait_for_drain
    session = CallSession(
        call_id="call-openai-greeting-tail",
        caller_channel_id="channel-openai-greeting-tail",
    )
    await engine.session_store.upsert_call(session)
    await engine.conversation_coordinator.on_tts_start(
        session.call_id,
        "stream:greeting",
    )
    await engine.conversation_coordinator.on_tts_start(
        session.call_id,
        "tts_segment:call-openai-greeting-tail",
    )

    await engine._note_provider_output_start(session.call_id)
    await engine._note_provider_output_end(
        session.call_id,
        session,
        clear_tts_gating_after_drain=True,
    )
    await asyncio.wait_for(drain_started.wait(), timeout=0.2)

    during = await engine.session_store.get_by_call_id(session.call_id)
    assert during.audio_capture_enabled is False
    assert during.tts_tokens == {
        "stream:greeting",
        "tts_segment:call-openai-greeting-tail",
    }

    # A later playback token must not be swept up by the deferred greeting
    # release if a new response starts while the drain observer is settling.
    await engine.conversation_coordinator.on_tts_start(
        session.call_id,
        "stream:next-response",
    )
    release_drain.set()
    await _wait_until(
        lambda: not engine._provider_output_drain_tasks.get(session.call_id)
    )

    after = await engine.session_store.get_by_call_id(session.call_id)
    assert after.audio_capture_enabled is False
    assert after.tts_tokens == {"stream:next-response"}
    greeting_provider.release_greeting_transport_guard.assert_awaited_once_with()

    await engine.conversation_coordinator.on_tts_end(
        session.call_id,
        "stream:next-response",
    )
    final = await engine.session_store.get_by_call_id(session.call_id)
    assert final.audio_capture_enabled is True
    assert final.tts_tokens == set()


@pytest.mark.asyncio
async def test_new_provider_audio_cancels_stale_drain_without_clearing_output_state():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine._call_bg_tasks = {}
    engine._provider_output_operations = {}
    engine._provider_output_drain_tasks = {}
    engine._agent_output_active_calls = set()
    engine.config = SimpleNamespace(audio_transport="externalmedia")
    engine.no_input_watchdog = SimpleNamespace(
        note_agent_output_start=AsyncMock(),
        note_agent_output_end=AsyncMock(),
    )
    engine._save_session = AsyncMock()
    drain_started = asyncio.Event()
    hold_drain = asyncio.Event()

    async def wait_for_drain(_call_id, **_kwargs):
        drain_started.set()
        await hold_drain.wait()
        return True

    engine._wait_for_call_audio_drain = wait_for_drain
    session = CallSession(
        call_id="call-overlapping-output",
        caller_channel_id="channel-overlapping-output",
    )
    await engine.session_store.upsert_call(session)

    await engine._note_provider_output_start(session.call_id)
    await engine._note_provider_output_end(session.call_id, session)
    await asyncio.wait_for(drain_started.wait(), timeout=0.2)
    stale_task = engine._provider_output_drain_tasks[session.call_id]

    await engine._note_provider_output_end(session.call_id, session)
    assert engine._provider_output_drain_tasks[session.call_id] is stale_task

    await engine._note_provider_output_start(session.call_id)
    await asyncio.sleep(0)

    assert stale_task.cancelled() or stale_task.done()
    assert session.call_id in engine._agent_output_active_calls
    assert session.call_id not in engine._provider_output_drain_tasks
    engine.no_input_watchdog.note_agent_output_end.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchdog_observer_failure_does_not_break_tts_gating():
    session_store = SessionStore()
    session = CallSession(
        call_id="call-observer-failure",
        caller_channel_id="channel-observer-failure",
    )
    await session_store.upsert_call(session)
    coordinator = ConversationCoordinator(session_store)
    coordinator.set_no_input_watchdog(
        SimpleNamespace(
            note_agent_output_start=AsyncMock(side_effect=RuntimeError("start failed")),
            note_agent_output_end=AsyncMock(side_effect=RuntimeError("end failed")),
        )
    )

    assert await coordinator.on_tts_start("call-observer-failure", "playback-1") is True
    during = await session_store.get_by_call_id("call-observer-failure")
    assert during.tts_playing is True
    assert during.audio_capture_enabled is False

    assert await coordinator.on_tts_end("call-observer-failure", "playback-1") is True
    after = await session_store.get_by_call_id("call-observer-failure")
    assert after.tts_playing is False
    assert after.audio_capture_enabled is True


@pytest.mark.asyncio
async def test_transport_gating_can_end_without_ending_provider_output_timing():
    session_store = SessionStore()
    session = CallSession(
        call_id="call-provider-drain-gating",
        caller_channel_id="channel-provider-drain-gating",
    )
    await session_store.upsert_call(session)
    watchdog = SimpleNamespace(
        note_agent_output_start=AsyncMock(),
        note_agent_output_end=AsyncMock(),
    )
    coordinator = ConversationCoordinator(session_store)
    coordinator.set_no_input_watchdog(watchdog)

    assert await coordinator.on_tts_start(session.call_id, "stream-1") is True
    assert await coordinator.on_tts_end(
        session.call_id,
        "stream-1",
        reason="provider-generation-complete",
        notify_no_input=False,
    ) is True

    watchdog.note_agent_output_start.assert_awaited_once_with(session.call_id)
    watchdog.note_agent_output_end.assert_not_awaited()
    after = await session_store.get_by_call_id(session.call_id)
    assert after.tts_playing is False
    assert after.audio_capture_enabled is True


@pytest.mark.asyncio
async def test_transfer_policy_callback_prevents_prompts_while_caller_is_on_hold():
    announcements = []
    paused = True

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    async def should_pause(call_id):
        return paused

    watchdog = NoInputWatchdog(announce, AsyncMock(), should_pause=should_pause)
    policy = NoInputPolicy(initial_timeout_sec=0.04, grace_timeout_sec=0.03, max_check_ins=1)
    await watchdog.register("call-hold", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("call-hold")
        await asyncio.sleep(0.1)
        assert announcements == []
        paused = False
        await _wait_until(lambda: announcements == ["check_in"])
    finally:
        await watchdog.stop("call-hold")


@pytest.mark.asyncio
async def test_outbound_calls_require_context_level_opt_in_even_if_global_is_true():
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(
        no_input=SimpleNamespace(
            model_dump=lambda: {
                "enabled": True,
                "inbound_enabled": True,
                "outbound_enabled": True,
                "initial_timeout_sec": 30,
                "grace_timeout_sec": 15,
                "max_check_ins": 1,
            }
        )
    )
    engine.no_input_watchdog = SimpleNamespace(register=AsyncMock())
    engine._save_session = AsyncMock()
    session = CallSession(
        call_id="outbound-1",
        caller_channel_id="channel-1",
        is_outbound=True,
    )

    await engine._configure_no_input_watchdog(session, SimpleNamespace(no_input={}))
    disabled_policy = engine.no_input_watchdog.register.await_args.args[1]
    assert disabled_policy.outbound_enabled is False

    await engine._configure_no_input_watchdog(
        session,
        SimpleNamespace(no_input={"outbound_enabled": "true", "initial_timeout_sec": 45}),
    )
    enabled_policy = engine.no_input_watchdog.register.await_args.args[1]
    assert enabled_policy.outbound_enabled is True
    assert enabled_policy.initial_timeout_sec == 45

    await engine._configure_no_input_watchdog(
        session,
        SimpleNamespace(no_input={"outbound_enabled": "false"}),
    )
    disabled_string_policy = engine.no_input_watchdog.register.await_args.args[1]
    assert disabled_string_policy.outbound_enabled is False


@pytest.mark.asyncio
async def test_engine_hangup_records_a_distinct_policy_outcome():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.conversation_coordinator = None
    engine.ari_client = SimpleNamespace(hangup_channel=AsyncMock())
    session = CallSession(call_id="silent-call", caller_channel_id="channel-silent")
    await engine.session_store.upsert_call(session)

    await engine._hangup_for_no_input("silent-call")

    updated = await engine.session_store.get_by_call_id("silent-call")
    assert updated.call_outcome == "no_input_timeout"
    assert updated.no_input_state["timed_out"] is True
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-silent")


@pytest.mark.asyncio
async def test_caller_audio_drain_waits_for_stream_buffers_and_quiet_tail():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine._provider_stream_queues = {}
    engine._provider_coalesce_buf = {}

    call_id = "buffered-announcement"
    await engine.session_store.upsert_call(
        CallSession(call_id=call_id, caller_channel_id="channel-buffered")
    )

    jitter_buffer = asyncio.Queue()
    jitter_buffer.put_nowait(b"audio")
    engine.streaming_playback_manager = SimpleNamespace(
        active_streams={
            call_id: {
                "buffered_bytes": 160,
                "last_real_emit_ts": None,
            }
        },
        jitter_buffers={call_id: jitter_buffer},
        frame_remainders={call_id: b"tail"},
    )

    drain_task = asyncio.create_task(
        engine._wait_for_call_audio_drain(
            call_id,
            timeout_sec=1.0,
            quiet_sec=0.03,
            reason="test",
        )
    )
    await asyncio.sleep(0.05)
    assert drain_task.done() is False

    engine.streaming_playback_manager.active_streams[call_id]["buffered_bytes"] = 0
    engine.streaming_playback_manager.active_streams[call_id]["last_real_emit_ts"] = time.time()
    jitter_buffer.get_nowait()
    engine.streaming_playback_manager.frame_remainders[call_id] = b""

    assert await drain_task is True


def test_terminal_quiet_tail_covers_audiosocket_and_externalmedia():
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(audio_transport="audiosocket")
    assert engine._terminal_transport_quiet_sec() == 0.35
    engine.config.audio_transport = "externalmedia"
    assert engine._terminal_transport_quiet_sec() == 0.5


@pytest.mark.asyncio
async def test_terminal_hangup_is_idempotent_and_uses_shared_drain():
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(audio_transport="audiosocket")
    engine.session_store = SessionStore()
    engine.conversation_coordinator = None
    engine.ari_client = SimpleNamespace(hangup_channel=AsyncMock())
    engine._wait_for_call_audio_drain = AsyncMock(return_value=True)
    session = CallSession(call_id="terminal-call", caller_channel_id="channel-terminal")
    await engine.session_store.upsert_call(session)

    assert await engine._terminate_call_after_audio(
        "terminal-call",
        reason="test",
        call_outcome="agent_hangup",
    ) is True
    assert await engine._terminate_call_after_audio("terminal-call", reason="duplicate") is False
    updated = await engine.session_store.get_by_call_id("terminal-call")
    assert updated.call_outcome == "agent_hangup"
    engine._wait_for_call_audio_drain.assert_awaited_once()
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-terminal")


@pytest.mark.asyncio
async def test_terminal_hangup_yields_to_transfer_state():
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(audio_transport="externalmedia")
    engine.session_store = SessionStore()
    engine.conversation_coordinator = None
    engine.ari_client = SimpleNamespace(hangup_channel=AsyncMock())
    engine._wait_for_call_audio_drain = AsyncMock(return_value=True)
    session = CallSession(call_id="transfer-call", caller_channel_id="channel-transfer")
    session.transfer_active = True
    await engine.session_store.upsert_call(session)

    assert await engine._terminate_call_after_audio("transfer-call", reason="test") is False
    engine._wait_for_call_audio_drain.assert_not_awaited()
    engine.ari_client.hangup_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_input_wait_keeps_gating_active_until_transport_drains(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.conversation_coordinator = ConversationCoordinator(engine.session_store)

    call_id = "gated-announcement"
    session = CallSession(call_id=call_id, caller_channel_id="channel-gated")
    session.tts_started_ts = 2.0
    session.tts_playing = False
    await engine.session_store.upsert_call(session)
    operation = engine._begin_provider_output_operation(
        call_id,
        "no-input:final:test",
        "no_input_final",
    )
    operation["audio_started"].set()
    operation["generation_done"].set()

    async def fake_drain(_call_id, **_kwargs):
        during = await engine.session_store.get_by_call_id(call_id)
        assert "no_input_drain:no-input:final:test" in during.tts_tokens
        assert during.tts_playing is True
        return True

    monkeypatch.setattr(engine, "_wait_for_call_audio_drain", fake_drain)

    assert await engine._wait_for_no_input_announcement(
        call_id,
        announcement_id="no-input:final:test",
        previous_tts_started_ts=1.0,
        timeout_sec=0.2,
    ) is True

    after = await engine.session_store.get_by_call_id(call_id)
    assert "no_input_drain:no-input:final:test" not in after.tts_tokens
    assert after.tts_playing is False


# --- the stall timer (no_input.stall_timeout_sec) -----------------------------------
#
# The check-ins count only while the caller is quiet, so hold music, noise or an
# IVR that keeps a speech detector busy never let them start, and for outbound
# calls they are off by default anyway; a call could run until the trunk dropped
# it. The stall timer counts from the last exchange (a caller turn reaching the
# model, an utterance the agent finished) whatever the line carries.


def test_stall_timeout_is_off_by_default_and_coerced_like_the_other_fields():
    assert NoInputPolicy().stall_timeout_sec == 0.0
    assert NoInputPolicy().stall_applies() is False
    assert NoInputPolicy.from_mapping({"stall_timeout_sec": "90"}).stall_timeout_sec == 90.0
    assert NoInputPolicy.from_mapping({"stall_timeout_sec": -5}).stall_timeout_sec == 0.0
    assert NoInputPolicy.from_mapping({"stall_timeout_sec": "soon"}).stall_timeout_sec == 0.0
    assert NoInputPolicy(stall_timeout_sec=90).stall_applies() is True
    # The block's master switch turns it off; the direction gates do not.
    assert NoInputPolicy(enabled=False, stall_timeout_sec=90).stall_applies() is False
    assert NoInputPolicy(outbound_enabled=False, stall_timeout_sec=90).stall_applies() is True
    assert NoInputConfig(stall_timeout_sec=90).stall_timeout_sec == 90.0
    with pytest.raises(ValueError):
        NoInputConfig(stall_timeout_sec=-1)


@pytest.mark.asyncio
async def test_stall_timer_hangs_up_while_the_line_carries_sound():
    announcements = []
    hangups = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, grace_timeout_sec=10, stall_timeout_sec=0.06)
    await watchdog.register("music", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("music")
        # Hold music: the detector reports the caller talking without a break.
        await watchdog.note_input_state("music", True, "engine:silero_vad")
        await asyncio.sleep(0.03)
        await watchdog.note_activity("music", "engine:silero_vad_barge_in")
        assert hangups == []
        await _wait_until(lambda: hangups == ["music"])
        assert announcements == ["final"]  # the final message is spoken before the hangup
        assert watchdog.snapshot("music")["phase"] == "stall_hangup"
    finally:
        await watchdog.stop("music")


@pytest.mark.asyncio
async def test_a_caller_turn_during_the_stall_final_message_keeps_the_call_alive():
    hangups = []
    phases_seen = []

    async def announce(call_id, text, kind):
        phases_seen.append(watchdog.snapshot(call_id)["phase"])
        if len(phases_seen) == 1:
            # The recognizer hands the model a caller turn while the final message plays.
            await watchdog.note_processing(call_id, True)
            await watchdog.note_processing(call_id, False)
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, grace_timeout_sec=10, stall_timeout_sec=0.06)
    await watchdog.register("late", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("late")
        await _wait_until(lambda: phases_seen == ["stall_announcement"])
        await asyncio.sleep(0.04)
        assert hangups == []
        assert watchdog.snapshot("late")["phase"] == "waiting"
        assert watchdog.has_call("late") is True
        # Nothing exchanged afterwards: the stall timer counts again from that turn and fires.
        await _wait_until(lambda: hangups == ["late"])
        assert phases_seen == ["stall_announcement", "stall_announcement"]
    finally:
        await watchdog.stop("late")


@pytest.mark.asyncio
async def test_stall_timer_runs_for_outbound_calls_without_check_ins():
    announcements = []
    hangups = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(initial_timeout_sec=0.02, grace_timeout_sec=0.02, stall_timeout_sec=0.06)
    assert await watchdog.register("outbound", policy, is_outbound=True) is True
    try:
        assert watchdog.snapshot("outbound")["inactivity_enabled"] is False
        await watchdog.mark_ready("outbound")
        await _wait_until(lambda: hangups == ["outbound"])
        assert announcements == ["final"]
    finally:
        await watchdog.stop("outbound")

    # Without either the check-ins or the stall timer nothing is registered.
    assert await watchdog.register("outbound-2", NoInputPolicy(), is_outbound=True) is False
    assert watchdog.has_call("outbound-2") is False


@pytest.mark.asyncio
async def test_exchanges_restart_the_stall_timer_but_caller_sound_does_not():
    hangups = []

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(AsyncMock(return_value=True), hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, stall_timeout_sec=0.06)
    await watchdog.register("talk", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("talk")
        await asyncio.sleep(0.04)
        await watchdog.note_processing("talk", True)  # a caller turn reached the model
        await asyncio.sleep(0.04)
        assert hangups == []
        await watchdog.note_agent_output_start("talk")  # the reply plays: paused
        await asyncio.sleep(0.08)
        assert hangups == []
        await watchdog.note_agent_output_end("talk")  # the reply finished: an exchange
        await asyncio.sleep(0.04)
        assert hangups == []
        await watchdog.note_input_state("talk", True, "engine:silero_vad")  # sound is not an exchange
        await _wait_until(lambda: hangups == ["talk"])
        assert watchdog.snapshot("talk")["last_exchange_source"] == "agent_output"
    finally:
        await watchdog.stop("talk")


@pytest.mark.asyncio
async def test_hosted_silence_output_does_not_restart_the_stall_timer():
    hangups = []

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(AsyncMock(return_value=True), hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, stall_timeout_sec=0.06)
    await watchdog.register("hosted", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("hosted")
        await asyncio.sleep(0.03)
        await watchdog.note_agent_output_start("hosted")
        await watchdog.note_agent_output_end("hosted", reset_timer=False)
        await asyncio.sleep(0.02)
        assert hangups == []
        # The deadline still counts from mark_ready.
        await _wait_until(lambda: hangups == ["hosted"], timeout=0.1)
    finally:
        await watchdog.stop("hosted")


@pytest.mark.asyncio
async def test_stall_timer_waits_while_the_caller_is_on_hold_or_in_a_transfer():
    hangups = []
    paused = {"value": True}

    async def hangup(call_id):
        hangups.append(call_id)

    async def should_pause(call_id):
        return paused["value"]

    watchdog = NoInputWatchdog(AsyncMock(return_value=True), hangup, should_pause=should_pause)
    policy = NoInputPolicy(initial_timeout_sec=10, stall_timeout_sec=0.04)
    await watchdog.register("hold", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("hold")
        await asyncio.sleep(0.1)
        assert hangups == []
        assert watchdog.snapshot("hold")["last_exchange_source"] == "policy_paused"
        paused["value"] = False
        await _wait_until(lambda: hangups == ["hold"])
    finally:
        await watchdog.stop("hold")


@pytest.mark.asyncio
async def test_the_check_ins_still_come_first_for_a_quiet_caller():
    announcements = []
    hangups = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(
        initial_timeout_sec=0.03, grace_timeout_sec=0.03, max_check_ins=1, stall_timeout_sec=0.5
    )
    await watchdog.register("quiet", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("quiet")
        await _wait_until(lambda: hangups == ["quiet"])
        assert announcements == ["check_in", "final"]
        assert watchdog.snapshot("quiet")["phase"] == "hangup"
    finally:
        await watchdog.stop("quiet")


@pytest.mark.asyncio
async def test_engine_records_why_the_watchdog_hung_up():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.conversation_coordinator = None
    engine.ari_client = SimpleNamespace(hangup_channel=AsyncMock())
    engine.no_input_watchdog = SimpleNamespace(
        snapshot=lambda call_id: {"phase": "stall_hangup", "stall_timeout_sec": 90.0, "last_exchange_source": "ready"}
    )
    session = CallSession(call_id="stalled-call", caller_channel_id="channel-stalled")
    await engine.session_store.upsert_call(session)

    await engine._hangup_for_no_input("stalled-call")

    updated = await engine.session_store.get_by_call_id("stalled-call")
    assert updated.call_outcome == "no_input_timeout"
    assert updated.no_input_state["timed_out_reason"] == "stall"
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-stalled")


# --- the hard duration cap (no_input.max_call_duration_sec) ---------------------------


def test_max_call_duration_is_off_by_default_and_coerced_like_the_other_fields():
    assert NoInputPolicy().max_call_duration_sec == 0.0
    assert NoInputPolicy().max_duration_applies() is False
    assert NoInputPolicy.from_mapping({"max_call_duration_sec": "1800"}).max_call_duration_sec == 1800.0
    assert NoInputPolicy.from_mapping({"max_call_duration_sec": -5}).max_call_duration_sec == 0.0
    assert NoInputPolicy.from_mapping({"max_call_duration_sec": 100000}).max_call_duration_sec == 0.0
    assert NoInputPolicy(max_call_duration_sec=1800).max_duration_applies() is True
    # The block's master switch turns it off; the direction gates do not.
    assert NoInputPolicy(enabled=False, max_call_duration_sec=1800).max_duration_applies() is False
    assert NoInputPolicy(outbound_enabled=False, max_call_duration_sec=1800).max_duration_applies() is True
    assert NoInputConfig(max_call_duration_sec=1800).max_call_duration_sec == 1800.0
    with pytest.raises(ValueError):
        NoInputConfig(max_call_duration_sec=-1)


@pytest.mark.asyncio
async def test_max_call_duration_hangs_up_whatever_the_call_is_doing():
    announcements = []
    hangups = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    policy = NoInputPolicy(initial_timeout_sec=10, grace_timeout_sec=10, max_call_duration_sec=0.06)
    # Outbound: neither the check-ins nor the stall timer apply; the cap alone registers.
    assert await watchdog.register("capped", policy, is_outbound=True) is True
    try:
        # No mark_ready: the cap does not wait for the greeting. The caller is
        # making sound and the agent is speaking: neither pauses it.
        await watchdog.note_input_state("capped", True, "engine:silero_vad")
        await watchdog.note_agent_output_start("capped")
        await asyncio.sleep(0.03)
        assert hangups == []
        await _wait_until(lambda: hangups == ["capped"])
        assert announcements == []
        assert watchdog.snapshot("capped")["phase"] == "max_duration_hangup"
    finally:
        await watchdog.stop("capped")


@pytest.mark.asyncio
async def test_max_call_duration_counts_from_the_calls_start():
    hangups = []

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(AsyncMock(return_value=True), hangup)
    started = time.monotonic()
    await watchdog.register("late", NoInputPolicy(max_call_duration_sec=0.2), is_outbound=False, elapsed_sec=0.15)
    try:
        await _wait_until(lambda: hangups == ["late"])
        assert time.monotonic() - started < 0.15
    finally:
        await watchdog.stop("late")


@pytest.mark.asyncio
async def test_max_call_duration_waits_for_a_transfer_to_finish(monkeypatch):
    import src.core.no_input_watchdog as watchdog_module

    monkeypatch.setattr(watchdog_module, "_HARD_LIMIT_RETRY_SEC", 0.02)
    hangups = []
    paused = {"value": True}

    async def hangup(call_id):
        hangups.append(call_id)

    async def should_pause(call_id):
        return paused["value"]

    watchdog = NoInputWatchdog(AsyncMock(return_value=True), hangup, should_pause=should_pause)
    await watchdog.register("transfer", NoInputPolicy(max_call_duration_sec=0.03), is_outbound=False)
    try:
        await asyncio.sleep(0.1)
        assert hangups == []
        paused["value"] = False
        await _wait_until(lambda: hangups == ["transfer"])
    finally:
        await watchdog.stop("transfer")


@pytest.mark.asyncio
async def test_the_check_ins_and_the_cap_coexist():
    announcements = []
    hangups = []

    async def announce(call_id, text, kind):
        announcements.append(kind)
        return True

    async def hangup(call_id):
        hangups.append(call_id)

    watchdog = NoInputWatchdog(announce, hangup)
    # A quiet caller still gets the check-in flow when the cap is far away...
    policy = NoInputPolicy(initial_timeout_sec=0.03, grace_timeout_sec=0.03, max_check_ins=1, max_call_duration_sec=1.0)
    await watchdog.register("quiet-capped", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("quiet-capped")
        await _wait_until(lambda: hangups == ["quiet-capped"])
        assert announcements == ["check_in", "final"]
        assert watchdog.snapshot("quiet-capped")["phase"] == "hangup"
    finally:
        await watchdog.stop("quiet-capped")

    # ...and the cap wins when it is nearer.
    announcements.clear()
    hangups.clear()
    policy = NoInputPolicy(initial_timeout_sec=0.5, grace_timeout_sec=0.5, max_check_ins=1, max_call_duration_sec=0.04)
    await watchdog.register("capped-first", policy, is_outbound=False)
    try:
        await watchdog.mark_ready("capped-first")
        await _wait_until(lambda: hangups == ["capped-first"])
        assert announcements == []
        assert watchdog.snapshot("capped-first")["phase"] == "max_duration_hangup"
    finally:
        await watchdog.stop("capped-first")


@pytest.mark.asyncio
async def test_engine_records_the_cap_as_its_own_outcome():
    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.conversation_coordinator = None
    engine.ari_client = SimpleNamespace(hangup_channel=AsyncMock())
    engine.no_input_watchdog = SimpleNamespace(
        snapshot=lambda call_id: {"phase": "max_duration_hangup", "max_call_duration_sec": 1800.0}
    )
    session = CallSession(call_id="long-call", caller_channel_id="channel-long")
    await engine.session_store.upsert_call(session)

    await engine._hangup_for_no_input("long-call")

    updated = await engine.session_store.get_by_call_id("long-call")
    assert updated.call_outcome == "max_duration"
    assert updated.no_input_state["timed_out_reason"] == "max_duration"
    engine.ari_client.hangup_channel.assert_awaited_once_with("channel-long")


@pytest.mark.asyncio
async def test_engine_registers_the_watchdog_with_the_time_the_call_already_lasted():
    from datetime import datetime, timedelta, timezone

    engine = Engine.__new__(Engine)
    engine.session_store = SessionStore()
    engine.conversation_coordinator = None
    engine.config = SimpleNamespace(no_input=NoInputConfig(max_call_duration_sec=1800))
    engine.no_input_watchdog = SimpleNamespace(register=AsyncMock(return_value=True))
    session = CallSession(call_id="amd-call", caller_channel_id="channel-amd")
    session.is_outbound = True
    session.start_time = datetime.now(timezone.utc) - timedelta(seconds=12)
    await engine.session_store.upsert_call(session)

    await engine._configure_no_input_watchdog(session, None)

    call = engine.no_input_watchdog.register.await_args
    assert call.kwargs["is_outbound"] is True
    assert 11.5 <= call.kwargs["elapsed_sec"] <= 13.0
    policy = call.args[1]
    assert policy.max_call_duration_sec == 1800.0
    assert policy.max_duration_applies() is True
    assert policy.applies_to(is_outbound=True) is False  # the check-ins stay opt-in for outbound
