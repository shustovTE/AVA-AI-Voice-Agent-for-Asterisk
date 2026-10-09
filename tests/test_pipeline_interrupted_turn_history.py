"""An interrupted streaming turn keeps the caller's words in the history.

The streaming overlap path appended the exchange only after the whole reply
had been spoken. A barge-in raised _PipelinePlaybackInterrupted before that
point and the turn returned with nothing recorded, so the model answered the
caller's next words with no memory of what they had just said — in a
production call it re-asked a question the caller had already answered.
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

CALL_ID = "call-interrupted"
CALLER_SAID = "я хочу полный капитальный ремонт у меня три комнаты"
SENTENCES = ["Первое предложение.", "Второе предложение.", "Третье предложение."]


def _config(*, heard_reply: bool = True, overlap: bool = True) -> AppConfig:
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
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {
                "pipeline_streaming_overlap": overlap,
                "pipeline_heard_reply_on_interrupt": heard_reply,
                "pipeline_heard_reply_lead_ms": 0,
            },
        }
    )


class _StreamingLLM(LLMComponent):
    supports_streaming = True

    async def generate(self, call_id, transcript, context, options):
        return " ".join(SENTENCES)

    async def generate_stream(self, call_id, transcript, context, options):
        for sentence in SENTENCES:
            for word in sentence.split(" "):
                yield word + " "


class _PlaybackStub:
    """Owns the stream; flipping ``active`` is what a barge-in does."""

    def __init__(self):
        # No stream until one is started, as with the real manager.
        self.active = False
        self.stopped = asyncio.Event()
        self.position_ms = 0  # what the hangup path reads as the played position

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.queue = queue
        self.active = True
        return "stream-1"

    def is_stream_active(self, call_id, stream_id=None):
        return self.active and (stream_id is None or stream_id == "stream-1")

    def get_playback_position_ms(self, call_id):
        return self.position_ms

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.active = False
        self.stopped.set()


class _InterruptingTTS(TTSComponent):
    """Cuts the stream while synthesizing the sentence at ``interrupt_at``."""

    downstream_mode_override = "stream"

    def __init__(self, playback, interrupt_at):
        self.playback = playback
        self.interrupt_at = interrupt_at
        self.synthesized = []

    async def synthesize(self, call_id, text, options):
        self.synthesized.append(text)
        if len(self.synthesized) == self.interrupt_at:
            self.playback.active = False
        yield b"\x00" * 320


async def _run_turn(monkeypatch, interrupt_at):
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    tts = _InterruptingTTS(playback, interrupt_at)
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=_StreamingLLM(),
        tts_adapter=tts,
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(
        engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution
    )
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=CALL_ID, caller_channel_id=CALL_ID)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)

    await stt.results.put(CALLER_SAID)
    await asyncio.wait_for(playback.stopped.wait(), timeout=3)
    await asyncio.sleep(0.1)
    updated = await engine.session_store.get_by_call_id(CALL_ID)
    history = [(m["role"], m["content"]) for m in (updated.conversation_history or [])]
    await engine._cleanup_call(CALL_ID)
    return history, tts


@pytest.mark.asyncio
async def test_interruption_mid_reply_keeps_the_caller_and_the_spoken_part(monkeypatch):
    history, tts = await _run_turn(monkeypatch, interrupt_at=2)

    assert ("user", CALLER_SAID) in history
    assert ("assistant", SENTENCES[0]) in history
    assert not any(SENTENCES[1] in content for _, content in history)
    assert not any(SENTENCES[2] in content for _, content in history)


@pytest.mark.asyncio
async def test_interruption_before_any_sentence_keeps_only_the_caller(monkeypatch):
    history, _ = await _run_turn(monkeypatch, interrupt_at=1)

    assert history == [("user", CALLER_SAID)]


@pytest.mark.asyncio
async def test_the_stream_was_actually_cut_short(monkeypatch):
    """Guards the harness: without an interruption all sentences are synthesized."""
    _, tts = await _run_turn(monkeypatch, interrupt_at=2)

    assert tts.synthesized == SENTENCES[:2]


# --- what the caller heard (streaming.pipeline_heard_reply_on_interrupt) -----------
#
# Every sentence above is 320 bytes of mu-law at 8 kHz: 40 ms of audio. The
# barge-in handler reports how much audio had reached the transport; the
# history then keeps the sentences that played in full plus a proportional
# prefix of the cut one, marked with an ellipsis.


class _HookingTTS(_InterruptingTTS):
    """Reports the barge-in position to the engine before cutting the stream, as the handler does."""

    engine = None
    played_ms = 0

    async def synthesize(self, call_id, text, options):
        self.synthesized.append(text)
        if len(self.synthesized) == self.interrupt_at:
            session = await self.engine.session_store.get_by_call_id(call_id)
            await self.engine._note_pipeline_reply_interrupted(session, self.played_ms)
            self.playback.active = False
        yield b"\x00" * 320


async def _wait_for_assistant(engine):
    for _ in range(60):
        session = await engine.session_store.get_by_call_id(CALL_ID)
        if any(m.get("role") == "assistant" for m in (session.conversation_history or [])):
            return session
        await asyncio.sleep(0.05)
    raise AssertionError("the turn never persisted an assistant entry")


async def _run_full_turn(monkeypatch, *, heard_reply=True, played_ms=60, owns_stream=True):
    """A turn whose audio is fully queued before the caller interrupts it."""
    engine = Engine(_config(heard_reply=heard_reply))
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    tts = _InterruptingTTS(playback, interrupt_at=0)
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=_StreamingLLM(),
        tts_adapter=tts,
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=CALL_ID, caller_channel_id=CALL_ID)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)

    await stt.results.put(CALLER_SAID)
    session = await _wait_for_assistant(engine)
    if not owns_stream:
        playback.is_stream_active = lambda call_id, stream_id=None: False
    record = await engine._note_pipeline_reply_interrupted(session, played_ms)
    playback.active = False
    updated = await engine.session_store.get_by_call_id(CALL_ID)
    history = list(updated.conversation_history or [])
    await engine._cleanup_call(CALL_ID)
    return history, record


@pytest.mark.asyncio
async def test_interruption_after_the_whole_reply_was_queued_keeps_only_the_heard_part(monkeypatch):
    history, record = await _run_full_turn(monkeypatch, played_ms=60)

    assistant = [m for m in history if m["role"] == "assistant"]
    assert [m["content"] for m in history if m["role"] == "user"] == [CALLER_SAID]
    assert len(assistant) == 1
    assert assistant[0]["content"] == "Первое предложение. Второе…"
    assert assistant[0]["interrupted"] is True
    assert record is not None and record.completed is True and record.played_ms == 60


@pytest.mark.asyncio
async def test_interruption_mid_synthesis_uses_the_heard_estimate(monkeypatch):
    """The barge-in lands while the second sentence is still being synthesized."""
    engine = Engine(_config())
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    tts = _HookingTTS(playback, interrupt_at=2)
    tts.engine = engine
    tts.played_ms = 60
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=_StreamingLLM(),
        tts_adapter=tts,
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=CALL_ID, caller_channel_id=CALL_ID)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)

    await stt.results.put(CALLER_SAID)
    await asyncio.wait_for(playback.stopped.wait(), timeout=3)
    await asyncio.sleep(0.1)
    updated = await engine.session_store.get_by_call_id(CALL_ID)
    history = list(updated.conversation_history or [])
    await engine._cleanup_call(CALL_ID)

    assert [m["content"] for m in history if m["role"] == "user"] == [CALLER_SAID]
    assistant = [m for m in history if m["role"] == "assistant"]
    assert len(assistant) == 1
    # Sentence one played in full (40 ms); sentence two, not yet synthesized, is
    # measured by sentence one's rate, so 20 ms into it is half a sentence.
    assert assistant[0]["content"] == "Первое предложение. Второе…"
    assert assistant[0]["interrupted"] is True
    assert tts.synthesized == SENTENCES[:2]


@pytest.mark.asyncio
async def test_the_switch_off_keeps_the_previous_behaviour(monkeypatch):
    history, record = await _run_full_turn(monkeypatch, heard_reply=False, played_ms=60)

    assert record is None
    assistant = [m for m in history if m["role"] == "assistant"]
    assert assistant[0]["content"] == " ".join(SENTENCES)
    assert "interrupted" not in assistant[0]


@pytest.mark.asyncio
async def test_a_cut_stream_that_is_not_the_reply_leaves_the_history_alone(monkeypatch):
    """A filler phrase or an announcement owns its own stream; cutting it must not rewrite the reply."""
    history, record = await _run_full_turn(monkeypatch, played_ms=60, owns_stream=False)

    assert record is None
    assistant = [m for m in history if m["role"] == "assistant"]
    assert assistant[0]["content"] == " ".join(SENTENCES)
    assert "interrupted" not in assistant[0]


# --- the trimmed reply must survive the next turn (serial mode) ---------------------
#
# The dialog worker kept its own copy of the history and persisted that copy
# after every turn, so a reply the barge-in handler had trimmed in the session
# came back whole at the next turn: the LLM saw the untrimmed offer and so did
# the call record, unless the caller hung up before another turn ran.


class _ContextLLM(LLMComponent):
    """Serial adapter that records the prior messages each turn was given."""

    supports_streaming = False

    def __init__(self):
        self.contexts = []
        self.replies = [" ".join(SENTENCES), "Ответ на второй вопрос."]

    async def generate(self, call_id, transcript, context, options):
        self.contexts.append(list(context.get("prior_messages") or []))
        return self.replies[min(len(self.contexts) - 1, len(self.replies) - 1)]


async def _wait_for_record(engine, call_id, *, completed=True):
    for _ in range(60):
        record = engine._spoken_replies.get(call_id)
        if record is not None and record.persisted_text and (record.completed or not completed):
            return record
        await asyncio.sleep(0.05)
    raise AssertionError("the reply's record never completed")


async def _start_serial(monkeypatch):
    # The engine's cleanup guard remembers finished call ids, so these tests,
    # which rely on the cleanup running, hang up a call of their own.
    call_id = f"call-serial-{uuid4().hex[:8]}"
    engine = Engine(_config(overlap=False))
    engine.pipeline_orchestrator._started = True
    playback = _PlaybackStub()
    monkeypatch.setattr(engine, "streaming_playback_manager", playback)
    stt = _ResultStreamingStubSTT()
    llm = _ContextLLM()
    resolution = _StubResolution(
        stt_adapter=stt,
        stt_options={"streaming": True, "chunk_ms": 80},
        llm_adapter=llm,
        tts_adapter=_InterruptingTTS(playback, interrupt_at=0),
    )
    resolution.llm_options = {"end_of_turn_silence_ms": 100}
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)

    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, session, playback, stt, llm


@pytest.mark.asyncio
async def test_the_trimmed_reply_survives_the_next_turn_and_reaches_the_llm(monkeypatch):
    engine, session, playback, stt, llm = await _start_serial(monkeypatch)
    await stt.results.put(CALLER_SAID)
    await _wait_for_record(engine, session.call_id)

    # The barge-in handler: 20 ms of the 40 ms reply had reached the transport.
    record = await engine._note_pipeline_reply_interrupted(session, 20)
    assert record is not None and record.interrupted
    trimmed = session.conversation_history[-1]
    assert trimmed["role"] == "assistant" and trimmed["interrupted"] is True
    assert trimmed["content"] != " ".join(SENTENCES) and trimmed["content"].endswith("…")

    await stt.results.put("а сколько это стоит")
    for _ in range(60):
        if len(llm.contexts) >= 2 and any(
            m.get("content") == "Ответ на второй вопрос." for m in session.conversation_history
        ):
            break
        await asyncio.sleep(0.05)
    history = [(m["role"], m["content"]) for m in session.conversation_history]
    await engine._cleanup_call(session.call_id)

    assert history == [
        ("user", CALLER_SAID),
        ("assistant", trimmed["content"]),
        ("user", "а сколько это стоит"),
        ("assistant", "Ответ на второй вопрос."),
    ]
    # The next turn's LLM saw only what the caller heard.
    assert [m["content"] for m in llm.contexts[1] if m["role"] == "assistant"] == [trimmed["content"]]


@pytest.mark.asyncio
async def test_a_hangup_during_the_reply_keeps_only_the_heard_part(monkeypatch):
    """The caller hanging up mid-reply is an interruption too: the record keeps what they heard."""
    engine, session, playback, stt, llm = await _start_serial(monkeypatch)
    await stt.results.put(CALLER_SAID)
    await _wait_for_record(engine, session.call_id)
    playback.position_ms = 20

    await engine._cleanup_call(session.call_id)

    history = [(m["role"], m["content"]) for m in session.conversation_history]
    assert history[0] == ("user", CALLER_SAID)
    assert len(history) == 2
    assistant = session.conversation_history[1]
    assert assistant["interrupted"] is True
    assert assistant["content"] != " ".join(SENTENCES) and assistant["content"].endswith("…")
