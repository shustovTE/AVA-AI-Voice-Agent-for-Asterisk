"""The stream memory, loudness normalization and FIR ingress of the VAD-gated offline recognizers.

A Silero VAD segment starts about 64 ms before the first window the VAD called
speech and ends where the closing silence began, and the pre-roll the server
kept was the tail of the stream, which at the moment a segment is closed is
the 700 ms of silence that closed it. Short phrases reached GigaAM v3 without
their first consonant, without their last one, at telephone level, through
an upsampler that mirrored the top of the band. The offline backends now keep
the recent stream by absolute position and widen every segment with the audio
that really surrounded it, bring it to a loudness target, and bring 8 kHz
clients to 16 kHz through a polyphase FIR.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

LOCAL_AI_DIR = str(Path(__file__).resolve().parents[1] / "local_ai_server")


def _load(name: str):
    if LOCAL_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_AI_DIR)
    return importlib.import_module(name)


def _pcm16(values) -> bytes:
    return np.asarray(values, dtype=np.int16).tobytes()


# --- the ring -------------------------------------------------------------------------


def test_the_context_reads_audio_by_absolute_position_and_pads_with_silence():
    sb = _load("stt_backends")
    context = sb.OfflineSegmentContext(sample_rate=16_000, capacity_seconds=0.001)  # 16 samples
    assert context.capacity == 16

    context.append(_pcm16(range(1, 11)))  # samples 0..9 hold 1..10
    assert context.total_samples == 10
    assert context.read(2, 5).tolist() == pytest.approx([3 / 32768, 4 / 32768, 5 / 32768])
    # Before the stream and past its end is silence.
    assert context.read(-3, 2).tolist() == pytest.approx([0.0, 0.0, 0.0, 1 / 32768, 2 / 32768])
    assert context.read(8, 13).tolist() == pytest.approx([9 / 32768, 10 / 32768, 0.0, 0.0, 0.0])
    assert len(context.read(5, 5)) == 0 and len(context.read(7, 3)) == 0

    context.append(_pcm16(range(11, 21)))  # wraps around: samples 10..19 hold 11..20
    assert context.total_samples == 20 and context.oldest_sample == 4
    assert context.read(3, 6).tolist() == pytest.approx([0.0, 5 / 32768, 6 / 32768])  # sample 3 was overwritten
    assert context.read(14, 20).tolist() == pytest.approx([v / 32768 for v in range(15, 21)])

    context.append(_pcm16(range(100, 140)))  # more than the ring holds: the last 16 samples survive
    assert context.total_samples == 60 and context.oldest_sample == 44
    assert context.read(44, 60).tolist() == pytest.approx([v / 32768 for v in range(124, 140)])
    assert context.read(40, 46).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 124 / 32768, 125 / 32768])


def test_the_context_maps_vad_sample_indices_onto_the_stream():
    sb = _load("stt_backends")
    context = sb.OfflineSegmentContext(sample_rate=16_000, capacity_seconds=1.0)
    context.bind_vad()
    assert context.absolute(800) == 800
    context.append(_pcm16([0] * 4000))
    context.bind_vad()  # the VAD created after a final starts at stream sample 4000
    assert context.vad_base_sample == 4000 and context.absolute(800) == 4800
    context.append(b"\x01")  # an odd trailing byte is not half a sample
    assert context.total_samples == 4000


def test_the_session_context_covers_the_longest_segment_and_its_context():
    sb = _load("stt_backends")
    backend = sb.SherpaOfflineSTTBackend(
        model_path="/fake/model",
        vad_model_path="/fake/vad.onnx",
        preroll_ms=350,
        postroll_ms=300,
        vad_min_silence_ms=700,
    )
    context = backend.create_session_context()
    assert isinstance(context, sb.OfflineSegmentContext)
    # 20 s of forced maximum + 0.7 s closing silence + 0.65 s of context + 1 s of slack.
    assert context.capacity == int(16_000 * (20.0 + 0.7 + 0.35 + 0.3 + 1.0))
    assert context.total_samples == 0 and context.vad_base_sample == 0


# --- normalization and the decode floor ----------------------------------------------


def _backend(**overrides):
    sb = _load("stt_backends")
    kwargs = dict(model_path="/fake/model", vad_model_path="/fake/vad.onnx")
    kwargs.update(overrides)
    return sb.SherpaOfflineSTTBackend(**kwargs)


def _dbfs(samples) -> float:
    return 20.0 * float(np.log10(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2))))


def test_normalization_targets_the_speech_part_and_never_clips():
    backend = _backend(normalize_dbfs=-20.0, normalize_max_gain_db=24.0)
    tone = 0.02 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16_000)  # about -37 dBFS
    widened = np.concatenate([np.zeros(1600), tone, np.zeros(800)]).astype(np.float32)

    out, gain_db = backend._normalize_segment(widened, slice(1600, 9600))

    assert gain_db == pytest.approx(-20.0 - _dbfs(tone), abs=0.01)
    assert _dbfs(out[1600:9600]) == pytest.approx(-20.0, abs=0.01)  # measured on the speech, not the padding
    assert np.all(out[:1600] == 0.0) and out.dtype == np.float32

    # A boost is capped; a hot segment is turned down; a peak is held below full scale.
    faint = (tone / 1000).astype(np.float32)
    _, gain_db = backend._normalize_segment(faint, slice(0, len(faint)))
    assert gain_db == pytest.approx(24.0)
    hot = (tone * 20).astype(np.float32)  # about -11 dBFS
    out, gain_db = backend._normalize_segment(hot, slice(0, len(hot)))
    assert gain_db < 0 and _dbfs(out) == pytest.approx(-20.0, abs=0.01)
    spiky = tone.astype(np.float32).copy()
    spiky[100] = 0.9
    out, gain_db = backend._normalize_segment(spiky, slice(0, len(spiky)))
    assert float(np.max(np.abs(out))) == pytest.approx(0.99, abs=1e-3) and 0 < gain_db < 17.0


def test_normalization_is_off_at_zero_and_leaves_silence_alone():
    off = _backend(normalize_dbfs=0)
    tone = (0.02 * np.sin(np.arange(4000) / 7.0)).astype(np.float32)
    out, gain_db = off._normalize_segment(tone, slice(0, 4000))
    assert out is tone and gain_db == 0.0
    assert _backend(normalize_dbfs="not-a-number").normalize_dbfs == 0.0
    assert _backend(normalize_dbfs=12).normalize_dbfs == 0.0  # a positive target makes no sense: off

    on = _backend(normalize_dbfs=-20)
    silence = np.zeros(4000, dtype=np.float32)
    out, gain_db = on._normalize_segment(silence, slice(0, 4000))
    assert out is silence and gain_db == 0.0
    assert on.tuning_summary().endswith("normalize_dbfs=-20.0 max_gain_db=24)")
    assert "normalize_dbfs=off" in off.tuning_summary()


def test_the_decode_floor_never_exceeds_the_vad_minimum_speech():
    assert _backend(vad_min_speech_ms=250)._min_audio_length == 4000  # the historical 250 ms
    assert _backend(vad_min_speech_ms=120)._min_audio_length == 1920  # follows a lowered minimum
    assert _backend(vad_min_speech_ms=0)._min_audio_length == 800  # but never below 50 ms


# --- the FIR ingress ------------------------------------------------------------------


def _tone(rate: int, hz: float, seconds: float = 0.5, amplitude: float = 10_000.0) -> np.ndarray:
    return np.rint(amplitude * np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate)).astype(np.int16)


def _band_energy_db(pcm: bytes, rate: int, low_hz: float, high_hz: float) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)[1024:]
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return 20.0 * np.log10(np.sqrt(np.sum(spectrum[band] ** 2)) / max(np.sqrt(np.sum(spectrum ** 2)), 1e-12))


def test_the_fir_upsampler_removes_the_images_that_ratecv_leaves():
    ap = _load("audio_processor")
    tone = _tone(8000, 3000.0)
    legacy = ap.AudioProcessor.resample_audio(tone.tobytes(), 8000, 16_000)
    fir = ap.FirUpsampler(8000, 16_000).process(tone.tobytes())

    assert len(fir) == 2 * len(tone) * 2
    # Linear interpolation mirrors a 3 kHz tone to 5 kHz at about -8 dB and drops the tone by 2-3 dB.
    assert _band_energy_db(legacy, 16_000, 4500, 8000) > -10.0
    assert _band_energy_db(fir, 16_000, 4500, 8000) < -80.0
    expected_rms = 10_000.0 / np.sqrt(2.0)
    fir_rms = np.sqrt(np.mean(np.frombuffer(fir, dtype=np.int16).astype(np.float64)[1024:] ** 2))
    assert abs(20.0 * np.log10(fir_rms / expected_rms)) < 0.1


def test_the_fir_upsampler_is_seamless_across_chunks():
    ap = _load("audio_processor")
    rng = np.random.default_rng(7)
    samples = rng.integers(-20_000, 20_001, size=2407, dtype=np.int16)
    pcm = samples.tobytes()
    one_shot = ap.FirUpsampler(8000, 16_000).process(pcm)

    streaming = ap.FirUpsampler(8000, 16_000)
    parts = []
    offset = 0
    for count in (17, 301, 2, 479, 91, 603, 914):
        parts.append(streaming.process(pcm[offset * 2 : (offset + count) * 2]))
        offset += count
    assert offset == len(samples)
    assert b"".join(parts) == one_shot
    assert streaming.process(b"") == b""

    streaming.reset()
    assert streaming.process(pcm) == one_shot
    tripled = ap.FirUpsampler(8000, 24_000).process(pcm)
    assert len(tripled) == 3 * len(pcm)


def test_the_fir_upsampler_takes_integer_ratios_only():
    ap = _load("audio_processor")
    with pytest.raises(ValueError):
        ap.FirUpsampler(16_000, 8000)
    with pytest.raises(ValueError):
        ap.FirUpsampler(8000, 11_025)
    with pytest.raises(ValueError):
        ap.FirUpsampler(8000, 8000)
