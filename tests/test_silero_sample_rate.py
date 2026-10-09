"""``vad.silero_sample_rate``: the rate Silero scores the caller's audio at.

After a silent line Silero's 8 kHz model put a caller's short answers ("тут",
"алло, алло?", "алло") right at the threshold, so whether they opened an
utterance depended on where its 32 ms chunk grid fell; on the recorded call
that lost them, its 16 kHz model, given the same audio upsampled, scored them
at 0.99-1.00 wherever the grid fell. Set to 16000, the engine upsamples 8 kHz
telephone audio for Silero alone: the recognizer, Smart Turn and the energy
checks keep the line's own audio, and Silero keeps its 32 ms cadence.
"""

import asyncio
import os

import numpy as np
import pytest
from pydantic import ValidationError

import src.core.silero_vad as sv
from src.audio.resampler import resample_audio
from src.config import VADConfig
from tests.test_pipeline_end_of_turn_silero import GRACE, _start_call
from tests.test_silero_protection_ends_with_reply import _capture_utterances


class _RecordingModel:
    """Answers with scripted probabilities and records what each run was given."""

    def __init__(self, probabilities=()):
        self.probabilities = list(probabilities)
        self.runs = []

    def run(self, samples, state, sample_rate):
        self.runs.append((int(sample_rate), int(samples.shape[-1])))
        probability = self.probabilities.pop(0) if self.probabilities else 0.0
        return probability, state


def _frames(rate: int, seconds: float):
    """20 ms frames of a steady tone, as a transport delivers them."""
    n = int(rate * seconds)
    pcm = (np.sin(np.arange(n) * 2 * np.pi * 440 / rate) * 3000).astype("<i2").tobytes()
    step = rate // 50 * 2
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


async def _feed(engine, session, frames, rate):
    for frame in frames:
        await engine._observe_silero_vad(session, frame, rate, source="test")


def test_the_setting_takes_8000_16000_or_nothing():
    assert VADConfig().silero_sample_rate is None
    assert VADConfig(silero_sample_rate=16000).silero_sample_rate == 16000
    assert VADConfig(silero_sample_rate=8000).silero_sample_rate == 8000
    with pytest.raises(ValidationError):
        VADConfig(silero_sample_rate=12000)


@pytest.mark.asyncio
async def test_unset_scores_the_line_at_its_own_rate(monkeypatch):
    model = _RecordingModel()
    engine, session, stt, llm = await _start_call(monkeypatch, "call-sr-line", GRACE, model)
    try:
        assert engine._silero_config()["sample_rate"] is None
        await _feed(engine, session, _frames(8000, 0.64), 8000)
        assert set(model.runs) == {(8000, 256 + 32)}      # Silero's 8 kHz chunk and context
        assert len(model.runs) == 20                      # one run per 32 ms
    finally:
        await engine._cleanup_call("call-sr-line")


@pytest.mark.asyncio
async def test_16000_scores_8k_telephone_audio_at_16k_on_the_same_cadence(monkeypatch):
    model = _RecordingModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch, "call-sr-16k", GRACE, model, vad={"silero_sample_rate": 16000}
    )
    try:
        await _feed(engine, session, _frames(8000, 0.64), 8000)
        assert set(model.runs) == {(16000, 512 + 64)}     # Silero's 16 kHz chunk and context
        assert len(model.runs) == 20                      # still one run per 32 ms
        assert "call-sr-16k" in engine._resample_state_silero_vad
    finally:
        await engine._cleanup_call("call-sr-16k")
    assert "call-sr-16k" not in engine._resample_state_silero_vad


@pytest.mark.asyncio
async def test_8000_scores_wideband_audio_at_8k(monkeypatch):
    model = _RecordingModel()
    engine, session, stt, llm = await _start_call(
        monkeypatch, "call-sr-8k", GRACE, model, vad={"silero_sample_rate": 8000}
    )
    try:
        await _feed(engine, session, _frames(16000, 0.64), 16000)
        assert {rate for rate, _ in model.runs} == {8000}
        assert len(model.runs) == 20
    finally:
        await engine._cleanup_call("call-sr-8k")


async def _utterance_heard(monkeypatch, call_id, vad):
    """One utterance cut by scripted probabilities from the same 8 kHz audio."""
    speech = [0.9] * 10 + [0.1] * 12
    model = _RecordingModel(speech)
    engine, session, stt, llm = await _start_call(
        monkeypatch, call_id, GRACE, model, vad={"silero_stt_utterances": True, "silero_stop_ms": 96, **vad}
    )
    try:
        sent = _capture_utterances(engine)
        await _feed(engine, session, _frames(8000, 32 * len(speech) / 1000.0), 8000)
        [utterance] = sent
        return utterance.pcm16, utterance.sample_rate
    finally:
        await engine._cleanup_call(call_id)


@pytest.mark.asyncio
async def test_the_recognizer_gets_the_line_audio_either_way(monkeypatch):
    at_line_rate = await _utterance_heard(monkeypatch, "call-sr-cut-line", {})
    at_16k = await _utterance_heard(monkeypatch, "call-sr-cut-16k", {"silero_sample_rate": 16000})
    assert at_16k == at_line_rate                       # same cut, same audio, the cutter's own 16 kHz


# --- the real graph, when available ---------------------------------------------

_REAL_MODEL = os.environ.get("SILERO_VAD_MODEL_PATH") or sv.DEFAULT_MODEL_PATH


@pytest.mark.skipif(
    not sv.ONNXRUNTIME_AVAILABLE or not os.path.isfile(_REAL_MODEL),
    reason="onnxruntime and the Silero VAD model file are needed",
)
def test_real_model_at_16k_still_scores_upsampled_silence_noise_and_tones_as_not_speech():
    model = sv.load_model(_REAL_MODEL)
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 8000.0
    signals = {
        "silence": np.zeros(16000),
        "noise": rng.standard_normal(16000) * 3000,
        "ringback": np.sin(2 * np.pi * 425 * t) * 8000,
    }
    for name, samples in signals.items():
        pcm = samples.astype("<i2").tobytes()
        state, upsampled = None, b""
        for i in range(0, len(pcm), 320):
            piece, state = resample_audio(pcm[i:i + 320], 8000, 16000, state=state, mode="linear")
            upsampled += piece
        assert max(sv.SileroVadStream(model, 16000).feed(upsampled)) < 0.3, name
