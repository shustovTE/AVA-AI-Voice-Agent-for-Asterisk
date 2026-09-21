"""The engine's local provider upsamples 8 kHz caller audio for the recognizer with a polyphase FIR.

The local provider brought the caller's 8 kHz audio to the 16 kHz the local
STT backends decode with linear interpolation, which rolls the telephone band
off by 2-3 dB and mirrors it above 4 kHz (a 3 kHz tone at 5 kHz, about -13 dB);
GigaAM v3 saw that mirrored energy in its mel features. ``stt_input_resampler``
on the provider block picks the alias-safe FIR (the default) or the legacy
linear interpolation.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from pydantic import ValidationError

from src.audio.resampler import resample_audio
from src.config import LocalProviderConfig
from src.providers.local import LocalProvider


class _FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.state = type("State", (), {"name": "OPEN"})()

    async def send(self, message):
        self.sent.append(message)


def test_the_provider_config_defaults_to_the_fir_and_takes_linear():
    assert LocalProviderConfig().stt_input_resampler == "fir"
    assert LocalProviderConfig(stt_input_resampler="linear").stt_input_resampler == "linear"
    with pytest.raises(ValidationError):
        LocalProviderConfig(stt_input_resampler="sox")


def test_the_provider_resolves_the_mode_from_its_config():
    assert LocalProvider(LocalProviderConfig(), on_event=None)._stt_input_resampler == "fir"
    assert LocalProvider(LocalProviderConfig(stt_input_resampler="linear"), on_event=None)._stt_input_resampler == "linear"


@pytest.mark.parametrize("mode", ["fir", "linear"])
@pytest.mark.asyncio
async def test_the_send_loop_upsamples_8khz_pcm_with_the_configured_resampler(mode):
    provider = LocalProvider(LocalProviderConfig(stt_input_resampler=mode), on_event=None)
    provider.input_mode = "pcm16_8k"
    provider._active_call_id = "call-1"
    provider.websocket = _FakeWebSocket()
    chunk = np.rint(8000 * np.sin(2 * np.pi * 1000 * np.arange(1600) / 8000)).astype(np.int16).tobytes()
    provider._send_queue.put_nowait(chunk)

    task = asyncio.create_task(provider._send_loop())
    try:
        for _ in range(200):
            if provider.websocket.sent:
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert len(provider.websocket.sent) == 1
    message = json.loads(provider.websocket.sent[0])
    assert message["type"] == "audio" and message["rate"] == 16000 and message["format"] == "pcm16le"
    expected, _ = resample_audio(chunk, 8000, 16000, mode=mode)
    assert base64.b64decode(message["data"]) == expected
    assert len(expected) == 2 * len(chunk)
