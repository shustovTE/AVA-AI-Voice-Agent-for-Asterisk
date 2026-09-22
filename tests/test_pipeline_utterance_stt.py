"""Silero cuts the caller's utterances for the recognizer (``vad.silero_stt_utterances``).

The recognizer no longer runs a voice activity detector of its own on a
continuous stream: the engine keeps the caller's audio and, the moment Silero
reports them quiet, sends everything from a little before their speech to its
end as one utterance. Nothing streams in between, no finalize burst is needed,
and a recognizer that takes only a stream still gets each utterance followed
by the silence its gate needs.
"""

import asyncio
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.config import AppConfig
from src.core.models import CallSession
from src.core.utterances import SttUtterance
from src.engine import Engine
from tests.test_pipeline_runner_lifecycle import _RecordingLLM, _ResultStreamingStubSTT, _StubResolution

CHUNK = b"\x11\x22" * 512  # one 32 ms Silero chunk at 16 kHz
GRACE = {"end_of_turn_talk_detect_grace_ms": 100}


class _ScriptedModel:
    def __init__(self, probabilities=()):
        self.probabilities = list(probabilities)

    def run(self, samples, state, sample_rate):
        probability = self.probabilities.pop(0) if self.probabilities else 0.0
        return probability, state


class _UtteranceStubSTT(_ResultStreamingStubSTT):
    """A recognizer that takes whole utterances, like the local AI server."""

    def __init__(self, supported=True):
        super().__init__()
        self.utterances = []
        self.supported = supported

    def utterances_supported(self, call_id):
        return self.supported

    async def send_utterance(self, call_id, audio, *, sample_rate_hz, utterance_id, fmt):
        self.utterances.append({"audio": audio, "rate": sample_rate_hz, "id": utterance_id, "fmt": fmt})


class _PlaybackStub:
    def __init__(self):
        self.active = False
        self.position_ms = 0
        self.active_streams = {}

    async def start_streaming_playback(self, call_id, queue, **kwargs):
        self.active = True
        self.active_streams[call_id] = {"stream_id": "stream-1", "playback_type": kwargs.get("playback_type")}
        return "stream-1"

    def is_stream_active(self, call_id, stream_id=None):
        return self.active

    def get_playback_position_ms(self, call_id):
        return self.position_ms

    async def stop_streaming_playback(self, call_id, *, drain=False):
        self.active = False
        self.active_streams.pop(call_id, None)
        return True


def _config(vad=None, barge_in=None, streaming=None) -> AppConfig:
    return AppConfig(
        **{
            "default_provider": "local",
            "providers": {"local": {"enabled": True}},
            "asterisk": {"host": "127.0.0.1", "port": 8088, "username": "u", "password": "p", "app_name": "a"},
            "llm": {"initial_greeting": "", "prompt": "You are helpful", "model": "gpt-4o"},
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "externalmedia",
            "downstream_mode": "stream",
            "streaming": {"pipeline_streaming_overlap": False, **(streaming or {})},
            "barge_in": {**(barge_in or {})},
            "vad": {
                "silero_enabled": True,
                "silero_start_ms": 96,
                "silero_stop_ms": 96,
                "silero_stt_utterances": True,
                "silero_utterance_preroll_ms": 64,
                **(vad or {}),
            },
        }
    )


async def _start_call(monkeypatch, *, stt, vad=None, barge_in=None, streaming=None):
    engine = Engine(_config(vad, barge_in, streaming))
    engine.pipeline_orchestrator._started = True
    model = _ScriptedModel()
    engine._silero_model = model
    monkeypatch.setattr(engine, "streaming_playback_manager", _PlaybackStub())
    llm = _RecordingLLM()
    resolution = _StubResolution(stt_adapter=stt, stt_options={"streaming": True, "chunk_ms": 80}, llm_adapter=llm)
    resolution.llm_options = dict(GRACE)
    monkeypatch.setattr(engine.pipeline_orchestrator, "get_pipeline", lambda *a, **k: resolution)
    engine.ari_client.set_channel_var = AsyncMock(return_value=True)
    call_id = f"call-utt-{uuid4().hex[:8]}"
    session = CallSession(call_id=call_id, caller_channel_id=call_id)
    session.pipeline_name = "streaming"
    session.conversation_state = "listening"
    session.audio_capture_enabled = True
    session.media_rx_confirmed = True
    await engine.session_store.upsert_call(session)
    await engine._ensure_pipeline_runner(session, forced=True)
    await asyncio.wait_for(stt.started.wait(), timeout=2)
    return engine, session, model, llm


async def _hear(engine, session, model, probabilities):
    model.probabilities.extend(probabilities)
    for _ in probabilities:
        await engine._observe_silero_vad(session, CHUNK, 16000, source="test")


async def _wait_for(predicate, timeout=2.0):
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
async def test_the_stop_sends_one_whole_utterance_and_nothing_streams(monkeypatch):
    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(monkeypatch, stt=stt)
    try:
        assert engine._pipeline_utterance_mode(session.call_id)
        await _hear(engine, session, model, [0.9, 0.9, 0.9])  # 96 ms: Silero start
        await _hear(engine, session, model, [0.1, 0.1, 0.1])  # 96 ms: Silero stop
        assert await _wait_for(lambda: len(stt.utterances) == 1)

        sent = stt.utterances[0]
        # Silero confirmed speech at the third chunk (1536 samples in); the utterance
        # starts 64 ms (1024 samples) before that and ends with the chunk that
        # closed it: samples 512..3072, i.e. 2560 samples.
        assert len(sent["audio"]) == 2560 * 2
        assert sent["rate"] == 16000 and sent["fmt"] == "pcm16_16k"
        assert sent["id"].startswith(f"{session.call_id}:utt-")
        assert stt.sent == []  # no raw frames, no finalize burst
        assert session.call_id in engine._pipeline_stt_final_expected_at

        await stt.results.put("алло")
        await asyncio.wait_for(llm.called.wait(), timeout=2)
        assert llm.transcripts == ["алло"]
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_recognizer_that_takes_only_a_stream_gets_the_utterance_with_a_closing_silence(monkeypatch):
    stt = _ResultStreamingStubSTT()  # no send_utterance
    engine, session, model, llm = await _start_call(monkeypatch, stt=stt)
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        assert await _wait_for(lambda: len(stt.sent) == 2)
        utterance, _ = stt.sent[0]
        silence, _ = stt.sent[1]
        assert len(utterance) == 5120
        assert len(silence) == 16000 * 2 * 1200 // 1000 and not any(silence)
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_an_unsupported_server_answer_switches_to_the_stream_fallback(monkeypatch):
    stt = _UtteranceStubSTT(supported=False)
    engine, session, model, llm = await _start_call(monkeypatch, stt=stt)
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        assert await _wait_for(lambda: len(stt.sent) == 2)
        assert stt.utterances == []
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_frames_while_the_agent_is_audible_are_muted_unless_listening_stays_on(monkeypatch):
    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(monkeypatch, stt=stt)
    try:
        playback = engine.streaming_playback_manager
        playback.active = True
        playback.position_ms = 400  # the reply is audible
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time()
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        await asyncio.sleep(0.1)
        assert stt.utterances == []  # all zeros: nothing worth decoding
        assert session.call_id not in engine._pipeline_stt_final_expected_at
    finally:
        await engine._cleanup_call(session.call_id)

    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(
        monkeypatch, stt=stt, barge_in={"pipeline_listen_during_playback": True}
    )
    try:
        playback = engine.streaming_playback_manager
        playback.active = True
        playback.position_ms = 400
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time()
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        assert await _wait_for(lambda: len(stt.utterances) == 1)
        assert stt.utterances[0]["audio"][-2:] == b"\x11\x22"  # the caller's audio, not zeros
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_before_the_first_sound_the_caller_is_not_muted(monkeypatch):
    """The gate closes at the stream start, but echo needs sound: until then the audio is real."""
    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(monkeypatch, stt=stt)
    try:
        playback = engine.streaming_playback_manager
        playback.active = True
        playback.position_ms = 0
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time()
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        assert await _wait_for(lambda: len(stt.utterances) == 1)
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_hangup_flushes_what_the_caller_was_saying(monkeypatch):
    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(
        monkeypatch, stt=stt, streaming={"pipeline_hangup_final_wait_ms": 200}
    )
    try:
        await _hear(engine, session, model, [0.9, 0.9, 0.9, 0.9, 0.9])  # still talking
        assert stt.utterances == []
        await engine._settle_pipeline_on_hangup(session)
        assert len(stt.utterances) == 1
        assert stt.utterances[0]["id"].endswith(":utt-1")
    finally:
        await engine._cleanup_call(session.call_id)


@pytest.mark.asyncio
async def test_a_caller_who_never_pauses_is_sent_in_pieces(monkeypatch):
    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(
        monkeypatch, stt=stt, vad={"silero_utterance_max_ms": 2000, "silero_utterance_preroll_ms": 0}
    )
    try:
        await _hear(engine, session, model, [0.9] * 70)  # 2.24 s of speech, no stop
        assert await _wait_for(lambda: len(stt.utterances) == 1)
        piece = stt.utterances[0]
        # The 2 s cap, reached at a chunk boundary: no quiet chunk to split at earlier.
        assert 16000 * 2 * 2 <= len(piece["audio"]) <= 16000 * 2 * 2 + 1024
        assert engine._utterance_cutters[session.call_id].open
        await _hear(engine, session, model, [0.1, 0.1, 0.1])
        assert await _wait_for(lambda: len(stt.utterances) == 2)
        assert stt.utterances[1]["id"].endswith(":utt-2")
    finally:
        await engine._cleanup_call(session.call_id)


def test_the_stt_queue_item_reports_its_length():
    item = SttUtterance(pcm16=b"\x00" * 3200, sample_rate=16000, utterance_id="u", started_at=0.0, ended_at=0.1)
    assert item.duration_ms == 100


@pytest.mark.asyncio
async def test_the_utterance_that_cuts_the_agent_off_reaches_the_recognizer_whole(monkeypatch):
    """Its head was spoken while the agent was audible: it is the caller's audio, not zeros."""
    stt = _UtteranceStubSTT()
    engine, session, model, llm = await _start_call(
        monkeypatch, stt=stt, barge_in={"talk_detect_initial_protection_ms": 0}
    )
    try:
        playback = engine.streaming_playback_manager
        playback.active = True
        playback.position_ms = 400  # the reply is audible: the caller's frames are muted
        session.audio_capture_enabled = False
        session.tts_playing = True
        session.tts_started_ts = time.time() - 2.0
        await _hear(engine, session, model, [0.9, 0.9, 0.9])  # Silero start: barge-in
        assert session.audio_capture_enabled is True
        await _hear(engine, session, model, [0.1, 0.1, 0.1])
        assert await _wait_for(lambda: len(stt.utterances) == 1)

        sent = stt.utterances[0]
        assert len(sent["audio"]) == 2560 * 2
        assert sent["audio"][:2] == b"\x11\x22"  # the gated head, as the caller said it
        assert 0 not in sent["audio"]
    finally:
        await engine._cleanup_call(session.call_id)
