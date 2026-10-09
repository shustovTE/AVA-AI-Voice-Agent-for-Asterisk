"""AudioSocket audio is taken little-endian, as it comes; the first-frame probe only logs.

The probe used to guess the byte order from the first frame's energy, and the
guess was backwards: byte-swapping a quiet, correctly ordered frame makes it
loud, so on a 16 kHz (``slin16``) call whose first frame was line noise or a
breath every frame of the call would have been swapped into noise. The rule
never ran only because the audio handler's own later ``import audioop`` made
the probe raise before it, which also meant the probe never logged.
"""

import audioop

import numpy as np
import pytest

import src.engine as engine_module
from src.audio.audiosocket_protocol import AudioSocketAudioFrame
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
            "pipelines": {"streaming": {}},
            "active_pipeline": "streaming",
            "audio_transport": "audiosocket",
        }
    )


def _zero_mean(samples: np.ndarray) -> bytes:
    """A 20 ms frame of the given samples followed by their negation: no DC to remove."""
    return np.concatenate([samples, -samples]).astype("<i2").tobytes()


def _line_noise(rate: int, level: int) -> bytes:
    rng = np.random.default_rng(7)
    return _zero_mean(np.clip(np.round(rng.normal(0, level, rate // 100)), -32767, 32767))


def _speech_like(rate: int) -> bytes:
    return _zero_mean(np.round(np.sin(np.arange(rate // 100) / 3.0) * 3000))


async def _call(monkeypatch):
    engine = Engine(_config())
    session = CallSession(call_id="call-bo", caller_channel_id="call-bo")
    await engine.session_store.upsert_call(session)
    engine.conn_to_channel["conn-bo"] = "call-bo"
    heard = []

    async def observe(session, pcm16, sample_rate_hz, *, source):
        heard.append((pcm16, sample_rate_hz))

    engine._observe_no_input_audio = observe
    probes = []
    real_info = engine_module.logger.info

    def info(event, *args, **kwargs):
        if event == "AudioSocket frame probe":
            probes.append(kwargs)
        return real_info(event, *args, **kwargs)

    monkeypatch.setattr(engine_module.logger, "info", info)
    return engine, heard, probes


async def _send(engine, payload: bytes, encoding: str, rate: int, message_type: int):
    await engine._audiosocket_handle_audio(
        "conn-bo", AudioSocketAudioFrame(payload, message_type, encoding, rate)
    )


@pytest.mark.asyncio
async def test_a_quiet_first_16k_frame_does_not_turn_the_call_into_noise(monkeypatch):
    engine, heard, _ = await _call(monkeypatch)
    first, then = _line_noise(16000, 40), _speech_like(16000)
    # The old rule would have called this frame byte-swapped and swapped every frame after it.
    swapped_rms = audioop.rms(audioop.byteswap(first, 2), 2)
    assert swapped_rms >= 2048 and swapped_rms >= 16 * audioop.rms(first, 2)

    await _send(engine, first, "slin16", 16000, 0x12)
    await _send(engine, then, "slin16", 16000, 0x12)

    assert heard == [(first, 16000), (then, 16000)]
    session = await engine.session_store.get_by_call_id("call-bo")
    assert session.vad_state["pcm16_inbound_swap"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding,rate,message_type",
    [("slin", 8000, 0x10), ("slin16", 16000, 0x12)],
)
async def test_the_probe_logs_the_first_frame_once_and_leaves_the_audio_alone(
    monkeypatch, encoding, rate, message_type
):
    engine, heard, probes = await _call(monkeypatch)
    frames = [_speech_like(rate), _line_noise(rate, 300)]
    for payload in frames:
        await _send(engine, payload, encoding, rate, message_type)

    assert [pcm for pcm, _ in heard] == frames
    [probe] = probes
    assert probe["sample_rate"] == rate
    assert probe["message_type"] == f"0x{message_type:02x}"
    assert probe["frame_bytes"] == len(frames[0])
    assert probe["rms_native"] == audioop.rms(frames[0], 2)
