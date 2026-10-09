"""The caller's next turn cuts a reply that is still playing.

A caller final that ended the turn while a pipeline reply was still on the
stream started a second reply on top of it. The playback manager handed the
second reply the live stream's id without taking its queue, so its audio was
never read: when the first reply ended the stream closed and the second was
discarded, while the history kept it as if spoken. The reply on the stream
was produced without the caller's latest words, so the turn now cuts it like
a barge-in (the history keeps the heard part) before answering, and the
manager replaces a live pipeline-tts stream instead of adopting a second
producer.
"""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.engine import Engine
from src.pipelines.base import LLMComponent, TTSComponent
from tests.test_pipeline_runner_lifecycle import _ResultStreamingStubSTT, _StubResolution

FIRST = "Первое предложение. Второе предложение. Третье предложение."
SECOND = "Ответ на второй вопрос."


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
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {
                "pipeline_streaming_overlap": False,
                "pipeline_heard_reply_on_interrupt": True,
                "pipeline_heard_reply_lead_ms": 0,
            },
        }
    )


class _TwoReplyLLM(LLMComponent):
    supports_streaming = False

    def __init__(self):
        self.contexts = []

    async def generate(self, call_id, transcript, context, options):
        self.contexts.append(list(context.get("prior_messages") or []))
        return FIRST if len(self.contexts) == 1 else SECOND


class _OneChunkTTS(TTSComponent):
    downstream_mode_override = "stream"

    async def synthesize(self, call_id, text, options):
        yield b"\x00" * 320  # 40 ms of mu-law: the whole reply


class _PlaybackStub:
    """A stream stays live until it is stopped, as the manager's pacer does while audio remains."""

    def __init__(self):
        self.active = False
        self.current = None
        self.starts = []
        self.stops = 0
        self.position_ms = 0
        self.active_streams = {}

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.current = f"stream-{len(self.starts) + 1}"
        self.starts.append(self.current)
        self.active = True
        self.active_streams[call_id] = {"stream_id": self.current, "playback_type": kwargs.get("playback_type")}
        return self.current

    def is_stream_active(self, call_id, stream_id=None):
        return self.active and (stream_id is None or stream_id == self.current)

    def get_playback_position_ms(self, call_id):
        return self.position_ms

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.stops += 1
        self.active = False
        self.active_streams.pop(call_id, None)
        return True


async def _wait_for(predicate, timeout=3.0):
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_the_next_turn_cuts_the_playing_reply_and_gets_a_stream_of_its_own(monkeypatch):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    llm = _TwoReplyLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=_OneChunkTTS(),
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    call_id = f"call-preempt-{uuid4().hex[:8]}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "serial"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)

    await stt.results.put("у меня три комнаты")
    assert await _wait_for(lambda: (record := engine._spoken_replies.get(call_id)) is not None and record.completed)
    assert playback.starts == ["stream-1"] and playback.active  # the first reply is still playing

    # The caller's next words arrive while it plays; half of its 40 ms had reached the transport.
    playback.position_ms = 20
    await stt.results.put("а сколько это стоит")
    assert await _wait_for(lambda: len(llm.contexts) == 2 and not playback.active or len(playback.starts) == 2)
    assert await _wait_for(lambda: any(m.get("content") == SECOND for m in session.conversation_history))
    history = [(m["role"], m["content"]) for m in session.conversation_history]
    await engine._cleanup_call(call_id)

    assert playback.stops >= 1 and playback.starts == ["stream-1", "stream-2"]
    assert history[0] == ("user", "у меня три комнаты")
    assert history[1][0] == "assistant" and history[1][1] != FIRST and history[1][1].endswith("…")
    assert session.conversation_history[1]["interrupted"] is True
    assert history[2:] == [("user", "а сколько это стоит"), ("assistant", SECOND)]
    # The second turn's model saw only what the caller heard of the first reply.
    assert [m["content"] for m in llm.contexts[1] if m["role"] == "assistant"] == [history[1][1]]


@pytest.mark.asyncio
async def test_nothing_is_cut_when_nothing_plays():
    engine = Engine(_config())
    playback = _PlaybackStub()
    engine.streaming_playback_manager = playback
    session = CallSession(call_id="quiet", caller_channel_id="quiet")

    assert await engine._cut_pipeline_playback_for_turn(session) is False
    assert playback.stops == 0


# --- the manager never hands a second pipeline reply a live stream ------------------


class _NeverDone:
    def done(self):
        return False


@pytest.mark.asyncio
async def test_the_manager_replaces_a_live_pipeline_stream_instead_of_reusing_it():
    from tests.test_continuous_stream import make_manager

    mgr = make_manager()
    call_id = "call-replace"
    mgr.active_streams[call_id] = {
        "stream_id": "stream:pipeline-tts:call-replace:1",
        "playback_type": "pipeline-tts",
        "streaming_task": _NeverDone(),
    }
    stopped = []

    async def _stop(cid, **kwargs):
        stopped.append(cid)
        mgr.active_streams.pop(cid, None)
        return True

    mgr.stop_streaming_playback = _stop
    # No session is registered, so after replacing the live stream the start
    # fails at the session lookup: what matters is that it did not reuse.
    result = await mgr.start_streaming_playback(call_id, asyncio.Queue(), playback_type="pipeline-tts")
    assert result is None
    assert stopped == [call_id]

    # A provider's continuous stream keeps being reused as before.
    mgr.active_streams[call_id] = {
        "stream_id": "stream:response:call-replace:2",
        "playback_type": "response",
        "streaming_task": _NeverDone(),
    }
    assert await mgr.start_streaming_playback(call_id, asyncio.Queue(), playback_type="response") == "stream:response:call-replace:2"
    assert stopped == [call_id]
