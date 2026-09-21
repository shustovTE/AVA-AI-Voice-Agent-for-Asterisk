from __future__ import annotations

import audioop
import io
import logging
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from constants import ULAW_SAMPLE_RATE

# Interpolation low-pass of ``FirUpsampler``: 48 taps per output phase and a
# Kaiser window (beta 8) give about 90 dB of image rejection with the cut-off
# at the source Nyquist frequency and a delay of about 3 ms at 16 kHz.
_FIR_UPSAMPLE_TAPS_PER_PHASE = 48
_FIR_UPSAMPLE_KAISER_BETA = 8.0
STT_RESAMPLER_MODES = ("fir", "ratecv")
DEFAULT_STT_RESAMPLER = "fir"


@lru_cache(maxsize=8)
def _fir_upsample_phases(factor: int) -> tuple:
    """Polyphase components of the interpolation low-pass for an integer ``factor``.

    The prototype is designed at the output rate with its cut-off at the source
    Nyquist frequency (``0.5 / factor`` cycles per output sample), Kaiser-windowed
    and scaled to a DC gain of ``factor`` to make up for the zero-stuffing it
    stands in for. Phase ``p`` holds every ``factor``-th tap from ``p`` on: output
    sample ``n * factor + p`` is its dot product with ``x[n], x[n-1], ...``.
    """
    n_taps = factor * _FIR_UPSAMPLE_TAPS_PER_PHASE
    center = (n_taps - 1) / 2.0
    positions = np.arange(n_taps, dtype=np.float64) - center
    cutoff = 0.5 / factor
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * positions)
    taps *= np.kaiser(n_taps, _FIR_UPSAMPLE_KAISER_BETA)
    taps *= factor / np.sum(taps)
    phases = []
    for phase in range(factor):
        component = np.ascontiguousarray(taps[phase::factor])
        component.setflags(write=False)
        phases.append(component)
    return tuple(phases)


class FirUpsampler:
    """Stateful polyphase windowed-sinc upsampler for an integer rate ratio.

    Brings 8 kHz caller audio to a recognizer's 16 kHz without the spectral
    images that the linear interpolation of ``audioop.ratecv`` leaves behind
    (a 3 kHz telephone tone mirrored to 5 kHz at about -8 dB, the band rolled
    off by 2-3 dB). It keeps the last taps of input between calls, so audio
    fed chunk by chunk comes out identical to the same audio fed at once.
    """

    def __init__(self, input_rate: int, output_rate: int):
        input_rate = int(input_rate)
        output_rate = int(output_rate)
        if input_rate <= 0 or output_rate <= input_rate or output_rate % input_rate:
            raise ValueError(
                f"FirUpsampler needs an integer upsampling ratio, got {input_rate} -> {output_rate}"
            )
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.factor = output_rate // input_rate
        self._phases = _fir_upsample_phases(self.factor)
        self._history = np.zeros(_FIR_UPSAMPLE_TAPS_PER_PHASE - 1, dtype=np.float64)

    def process(self, pcm16_audio: bytes) -> bytes:
        """Upsample one chunk of mono PCM16; returns ``factor`` times as many samples."""
        if not pcm16_audio:
            return b""
        audio = np.frombuffer(pcm16_audio[: len(pcm16_audio) & ~1], dtype=np.int16).astype(np.float64)
        if len(audio) == 0:
            return b""
        extended = np.concatenate((self._history, audio))
        resampled = np.empty(len(audio) * self.factor, dtype=np.float64)
        for phase, taps in enumerate(self._phases):
            # ``valid`` with the history prepended yields one output per source
            # sample, each looking back over the previous ``taps`` source samples.
            resampled[phase::self.factor] = np.convolve(extended, taps, mode="valid")
        self._history = extended[-len(self._history):].copy()
        return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16).tobytes()

    def reset(self) -> None:
        self._history[:] = 0.0


@dataclass(frozen=True)
class SynthesizedAudio:
    """TTS bytes with the format metadata required by downstream playback."""

    data: bytes
    encoding: str = "mulaw"
    sample_rate_hz: int = ULAW_SAMPLE_RATE

    def __bool__(self) -> bool:
        return bool(self.data)


class AudioProcessor:
    """Handles audio format conversions for MVP uLaw 8kHz pipeline.

    Primary path uses Python's audioop (in-process, no temp files).
    Falls back to sox subprocess if audioop conversion fails.
    """

    @staticmethod
    def resample_audio(
        input_data: bytes,
        input_rate: int,
        output_rate: int,
        input_format: str = "raw",
        output_format: str = "raw",
    ) -> bytes:
        """Resample raw PCM16 mono audio in-process via audioop.

        Falls back to sox subprocess on failure.
        """
        if input_rate == output_rate:
            return input_data

        try:
            resampled, _ = audioop.ratecv(input_data, 2, 1, input_rate, output_rate, None)
            return resampled
        except Exception as exc:
            logging.warning("audioop resample failed (%s), falling back to sox", exc)

        # ── sox fallback ──
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{input_format}", delete=False
            ) as input_file:
                input_file.write(input_data)
                input_path = input_file.name

            with tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ) as output_file:
                output_path = output_file.name

            cmd = [
                "sox",
                "-t", "raw",
                "-r", str(input_rate),
                "-e", "signed-integer",
                "-b", "16",
                "-c", "1",
                input_path,
                "-r", str(output_rate),
                "-c", "1",
                "-e", "signed-integer",
                "-b", "16",
                output_path,
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            with open(output_path, "rb") as f:
                resampled_data = f.read()

            os.unlink(input_path)
            os.unlink(output_path)

            return resampled_data

        except Exception as exc:  # pragma: no cover
            logging.error("Audio resampling failed (sox fallback): %s", exc)
            return input_data

    @staticmethod
    def pcm16_to_ulaw_8k(pcm_data: bytes, input_rate: int) -> bytes:
        """Convert raw PCM16 mono audio to 8 kHz µ-law in-process.

        Skips WAV header parsing — use when you already have raw PCM16 bytes.
        Falls back to convert_to_ulaw_8k via a WAV wrapper on failure.
        """
        # Preserve original data for fallback (ratecv reassigns pcm_data)
        original_pcm = pcm_data
        try:
            if input_rate != ULAW_SAMPLE_RATE:
                pcm_data, _ = audioop.ratecv(
                    pcm_data, 2, 1, input_rate, ULAW_SAMPLE_RATE, None
                )
            return audioop.lin2ulaw(pcm_data, 2)
        except Exception as exc:
            logging.warning(
                "pcm16_to_ulaw_8k failed (%s), falling back to WAV path", exc
            )
            # Build a minimal WAV wrapper using ORIGINAL data and delegate
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(input_rate)
                wf.writeframes(original_pcm)
            ulaw_data = AudioProcessor.convert_to_ulaw_8k(buf.getvalue(), input_rate)
            if not ulaw_data:
                logging.error("pcm16_to_ulaw_8k fallback also failed, returning empty")
                return b""
            return ulaw_data

    @staticmethod
    def convert_to_ulaw_8k(input_data: bytes, input_rate: int) -> bytes:
        """Convert WAV audio to 8 kHz µ-law in-process via audioop.

        Falls back to sox subprocess on failure.
        """
        try:
            # Parse WAV header to extract raw PCM
            wav_io = io.BytesIO(input_data)
            with wave.open(wav_io, "rb") as wf:
                pcm_data = wf.readframes(wf.getnframes())
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()

            # Stereo → mono
            if channels > 1:
                pcm_data = audioop.tomono(pcm_data, sampwidth, 1, 1)

            # Ensure 16-bit samples
            if sampwidth != 2:
                pcm_data = audioop.lin2lin(pcm_data, sampwidth, 2)

            # Resample to 8 kHz
            if framerate != ULAW_SAMPLE_RATE:
                pcm_data, _ = audioop.ratecv(
                    pcm_data, 2, 1, framerate, ULAW_SAMPLE_RATE, None
                )

            # PCM16 → µ-law
            ulaw_data = audioop.lin2ulaw(pcm_data, 2)
            return ulaw_data

        except Exception as exc:
            logging.warning(
                "audioop uLaw conversion failed (%s), falling back to sox", exc
            )

        # ── sox fallback ──
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
                input_file.write(input_data)
                input_path = input_file.name

            with tempfile.NamedTemporaryFile(suffix=".ulaw", delete=False) as output_file:
                output_path = output_file.name

            cmd = [
                "sox",
                input_path,
                "-r", str(ULAW_SAMPLE_RATE),
                "-c", "1",
                "-e", "mu-law",
                "-t", "raw",
                output_path,
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            with open(output_path, "rb") as f:
                ulaw_data = f.read()

            os.unlink(input_path)
            os.unlink(output_path)

            return ulaw_data

        except Exception as exc:  # pragma: no cover
            logging.error("uLaw conversion failed (sox fallback): %s", exc)
            return input_data
