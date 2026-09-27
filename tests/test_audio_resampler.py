import audioop
import struct

import numpy as np
import pytest

from src.audio import (
    StreamingResampler,
    convert_pcm16le_to_target_format,
    mulaw_to_pcm16le,
    pcm16le_to_mulaw,
    resample_audio,
    resolve_output_resampler_policy,
    resolve_stt_input_resampler,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ("linear", "profile")),
        ({"profile_mode": "bandlimited"}, ("bandlimited", "profile")),
        (
            {"profile_mode": "bandlimited", "provider_mode": "linear"},
            ("linear", "provider"),
        ),
        (
            {
                "profile_mode": "linear",
                "provider_mode": "bandlimited",
                "pipeline_mode": "linear",
            },
            ("linear", "pipeline"),
        ),
        (
            {
                "profile_mode": "linear",
                "provider_mode": "linear",
                "pipeline_mode": "linear",
                "environment_mode": "bandlimited",
            },
            ("bandlimited", "environment"),
        ),
        (
            {
                "profile_mode": "bandlimited",
                "provider_mode": "inherit",
                "pipeline_mode": "inherit",
            },
            ("bandlimited", "profile"),
        ),
        (
            {"profile_mode": "bandlimited", "environment_mode": "invalid"},
            ("linear", "environment-invalid-fallback"),
        ),
    ],
)
def test_output_resampler_policy_precedence(kwargs, expected):
    assert resolve_output_resampler_policy(**kwargs) == expected


def test_mulaw_round_trip_identity():
    pcm_samples = audioop.tostereo(b"\x00\x10" * 40, 2, 1, 1)  # create dummy PCM16 data
    mono_pcm = audioop.tomono(pcm_samples, 2, 1, 0)
    mulaw = pcm16le_to_mulaw(mono_pcm)
    restored = mulaw_to_pcm16le(mulaw)
    assert len(restored) == len(mono_pcm)
    restored_rms = audioop.rms(restored, 2)
    original_rms = audioop.rms(mono_pcm, 2)
    assert restored_rms == pytest.approx(
        original_rms, abs=8
    )


def test_resample_identity_when_rates_match():
    pcm = b"\x01\x02" * 160
    converted, state = resample_audio(pcm, 8000, 8000)
    assert converted == pcm
    assert state is None


def test_convert_pcm_to_ulaw_format():
    pcm = b"\x01\x02" * 160
    ulaw = convert_pcm16le_to_target_format(pcm, "ulaw")
    assert len(ulaw) == len(pcm) // 2  # μ-law is 1 byte per sample


# ── NumPy resampler: exact output sizing ──────────────────────────────


def test_upsample_8k_to_16k_exact_size():
    """160 samples @ 8kHz → 320 samples @ 16kHz = 640 bytes."""
    pcm_8k = b"\x00\x01" * 160  # 160 samples = 320 bytes
    out, _ = resample_audio(pcm_8k, 8000, 16000)
    assert len(out) == 640


def test_downsample_24k_to_8k_exact_size():
    """480 samples @ 24kHz (20 ms) → 160 samples @ 8kHz = 320 bytes."""
    pcm_24k = b"\x00\x01" * 480  # 480 samples = 960 bytes
    out, _ = resample_audio(pcm_24k, 24000, 8000)
    assert len(out) == 320


def test_bandlimited_downsample_24k_to_8k_exact_size():
    """The opt-in FIR path preserves the 20 ms telephony frame duration."""
    pcm_24k = b"\x00\x01" * 480
    out, state = resample_audio(pcm_24k, 24000, 8000, mode="bandlimited")
    assert len(out) == 320
    assert state is not None


def test_upsample_8k_to_24k_exact_size():
    """160 samples @ 8kHz → 480 samples @ 24kHz = 960 bytes."""
    pcm_8k = b"\x00\x01" * 160
    out, _ = resample_audio(pcm_8k, 8000, 24000)
    assert len(out) == 960


# ── NumPy resampler: state continuity ─────────────────────────────────


def test_state_continuity_across_chunks():
    """Resample two consecutive chunks with state carry and verify smooth boundary."""
    import math

    # Generate a 440 Hz sine wave at 8 kHz, two 20 ms chunks
    freq, sr = 440, 8000
    chunk_samples = 160  # 20 ms @ 8 kHz
    total_samples = chunk_samples * 2

    samples = [int(16000 * math.sin(2 * math.pi * freq * i / sr)) for i in range(total_samples)]
    chunk_a = struct.pack(f"<{chunk_samples}h", *samples[:chunk_samples])
    chunk_b = struct.pack(f"<{chunk_samples}h", *samples[chunk_samples:])

    # Resample chunk A, then chunk B with state carry
    out_a, state = resample_audio(chunk_a, 8000, 16000)
    out_b, _ = resample_audio(chunk_b, 8000, 16000, state=state)

    # Decode boundary samples
    sa = struct.unpack_from("<h", out_a, len(out_a) - 2)[0]
    sb = struct.unpack_from("<h", out_b, 0)[0]

    # The boundary jump should be small (smooth interpolation, not a click/pop)
    assert abs(sb - sa) < 2000, f"Boundary discontinuity too large: {abs(sb - sa)}"


def test_stateful_and_stateless_produce_valid_equal_length_output():
    """Stateful and stateless resampling both produce valid output of equal length."""
    pcm = b"\x10\x00" * 320  # 320 samples
    chunk_a = pcm[:320]
    chunk_b = pcm[320:]

    # Stateful path
    _, state = resample_audio(chunk_a, 8000, 16000)
    out_stateful, _ = resample_audio(chunk_b, 8000, 16000, state=state)

    # Stateless path (state=None)
    out_stateless, _ = resample_audio(chunk_b, 8000, 16000)

    # They may or may not differ depending on input, but both should produce valid output
    assert len(out_stateful) == len(out_stateless)


def test_bandlimited_streaming_matches_one_shot_for_irregular_chunks():
    """FIR history and decimation phase must make chunking sample-exact."""
    rng = np.random.default_rng(20260721)
    samples = rng.integers(-20000, 20001, size=2407, dtype=np.int16)
    pcm = samples.astype("<i2", copy=False).tobytes()

    one_shot, _ = resample_audio(pcm, 24000, 8000, mode="bandlimited")

    state = None
    chunks = []
    offset = 0
    for sample_count in (17, 301, 2, 479, 91, 603, 914):
        chunk = pcm[offset * 2 : (offset + sample_count) * 2]
        converted, state = resample_audio(
            chunk,
            24000,
            8000,
            state=state,
            mode="bandlimited",
        )
        chunks.append(converted)
        offset += sample_count

    assert offset == len(samples)
    assert b"".join(chunks) == one_shot


def test_bandlimited_downsample_suppresses_above_nyquist_alias():
    """A 5 kHz provider tone must not fold into the 8 kHz phone band at 3 kHz."""
    source_rate = 24000
    target_rate = 8000
    duration_samples = source_rate // 2
    positions = np.arange(duration_samples, dtype=np.float64)
    tone = np.rint(
        20000.0 * np.sin(2.0 * np.pi * 5000.0 * positions / source_rate)
    ).astype("<i2")

    legacy, _ = resample_audio(tone.tobytes(), source_rate, target_rate)
    filtered, _ = resample_audio(
        tone.tobytes(), source_rate, target_rate, mode="bandlimited"
    )
    # Ignore causal FIR startup while its zero-filled history is flushed.
    legacy_samples = np.frombuffer(legacy, dtype="<i2").astype(np.float64)[256:]
    filtered_samples = np.frombuffer(filtered, dtype="<i2").astype(np.float64)[256:]
    legacy_rms = np.sqrt(np.mean(np.square(legacy_samples)))
    filtered_rms = np.sqrt(np.mean(np.square(filtered_samples)))

    attenuation_db = 20.0 * np.log10(max(filtered_rms, 1e-12) / legacy_rms)
    assert attenuation_db < -60.0


def test_bandlimited_downsample_preserves_telephone_passband_tone():
    """A 3 kHz tone remains within 1 dB after alias-safe conversion."""
    source_rate = 24000
    target_rate = 8000
    duration_samples = source_rate // 2
    positions = np.arange(duration_samples, dtype=np.float64)
    tone = np.rint(
        20000.0 * np.sin(2.0 * np.pi * 3000.0 * positions / source_rate)
    ).astype("<i2")

    filtered, _ = resample_audio(
        tone.tobytes(), source_rate, target_rate, mode="bandlimited"
    )
    filtered_samples = np.frombuffer(filtered, dtype="<i2").astype(np.float64)[256:]
    filtered_rms = np.sqrt(np.mean(np.square(filtered_samples)))
    expected_rms = 20000.0 / np.sqrt(2.0)
    gain_db = 20.0 * np.log10(filtered_rms / expected_rms)

    assert abs(gain_db) < 1.0


def test_bandlimited_mode_falls_back_to_legacy_for_upsampling():
    """The candidate must not change inbound/upstream sample-rate conversion."""
    pcm = np.arange(-160, 160, dtype="<i2").tobytes()
    legacy, legacy_state = resample_audio(pcm, 8000, 24000)
    candidate, candidate_state = resample_audio(
        pcm, 8000, 24000, mode="bandlimited"
    )
    assert candidate == legacy
    assert candidate_state == legacy_state


# ── FIR upsampling (the STT ingress) ─────────────────────────────────


def _band_energy_db(pcm: bytes, rate: int, low_hz: float, high_hz: float) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)[1024:]
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return 20.0 * np.log10(np.sqrt(np.sum(spectrum[band] ** 2)) / max(np.sqrt(np.sum(spectrum ** 2)), 1e-12))


def _tone(rate: int, hz: float, seconds: float = 0.5) -> bytes:
    positions = np.arange(int(rate * seconds), dtype=np.float64)
    return np.rint(10000.0 * np.sin(2.0 * np.pi * hz * positions / rate)).astype("<i2").tobytes()


def test_fir_upsample_8k_to_16k_exact_size_and_24k():
    pcm = np.arange(-160, 160, dtype="<i2").tobytes()
    out, state = resample_audio(pcm, 8000, 16000, mode="fir")
    assert len(out) == 2 * len(pcm)
    assert state[0] == "fir_upsample_v1"
    out, _ = resample_audio(pcm, 8000, 24000, mode="fir")
    assert len(out) == 3 * len(pcm)


def test_fir_upsample_removes_the_images_linear_interpolation_leaves():
    """A 3 kHz phone tone must not come back mirrored at 5 kHz, and must keep its level."""
    tone = _tone(8000, 3000.0)
    legacy, _ = resample_audio(tone, 8000, 16000)
    fir, _ = resample_audio(tone, 8000, 16000, mode="fir")

    assert _band_energy_db(legacy, 16000, 4500, 8000) > -10.0  # the image linear interpolation leaves
    assert _band_energy_db(fir, 16000, 4500, 8000) < -80.0
    for hz in (300.0, 1000.0, 3000.0, 3400.0):
        out, _ = resample_audio(_tone(8000, hz), 8000, 16000, mode="fir")
        rms = np.sqrt(np.mean(np.frombuffer(out, dtype="<i2").astype(np.float64)[1024:] ** 2))
        assert abs(20.0 * np.log10(rms / (10000.0 / np.sqrt(2.0)))) < 0.1


def test_fir_upsample_streaming_matches_one_shot_for_irregular_chunks():
    rng = np.random.default_rng(20260921)
    samples = rng.integers(-20000, 20001, size=2407, dtype=np.int16)
    pcm = samples.astype("<i2", copy=False).tobytes()
    one_shot, _ = resample_audio(pcm, 8000, 16000, mode="fir")

    state = None
    chunks = []
    offset = 0
    for sample_count in (17, 301, 2, 479, 91, 603, 914):
        converted, state = resample_audio(
            pcm[offset * 2 : (offset + sample_count) * 2], 8000, 16000, state=state, mode="fir"
        )
        chunks.append(converted)
        offset += sample_count

    assert offset == len(samples)
    assert b"".join(chunks) == one_shot


def test_fir_mode_downsamples_like_bandlimited_and_falls_back_to_linear_otherwise():
    pcm = _tone(24000, 1000.0)
    fir, _ = resample_audio(pcm, 24000, 8000, mode="fir")
    bandlimited, _ = resample_audio(pcm, 24000, 8000, mode="bandlimited")
    assert fir == bandlimited

    pcm = np.arange(-160, 160, dtype="<i2").tobytes()
    fir, fir_state = resample_audio(pcm, 8000, 11025, mode="fir")
    linear, linear_state = resample_audio(pcm, 8000, 11025)
    assert fir == linear and fir_state == linear_state
    # A stale FIR state handed to the linear path is ignored, not misread as a sample.
    out, _ = resample_audio(pcm, 8000, 16000, state=("fir_upsample_v1", 8000, 16000, np.zeros(47)))
    assert out == resample_audio(pcm, 8000, 16000)[0]


def test_stt_input_resampler_resolution():
    assert resolve_stt_input_resampler(None) == "fir"
    assert resolve_stt_input_resampler("") == "fir"
    assert resolve_stt_input_resampler(" Linear ") == "linear"
    assert resolve_stt_input_resampler("bandlimited") == "fir"


# ── Edge cases ────────────────────────────────────────────────────────


def test_empty_input_returns_empty():
    out, state = resample_audio(b"", 8000, 16000)
    assert out == b""
    assert state is None


def test_resample_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported resample mode"):
        resample_audio(b"\x00\x00" * 20, 24000, 8000, mode="unknown")


def test_single_sample_upsample():
    """A single sample (2 bytes) should produce a valid output."""
    pcm = struct.pack("<h", 1000)
    out, state = resample_audio(pcm, 8000, 16000)
    assert len(out) == 4  # 1 sample @ 8k → 2 samples @ 16k = 4 bytes
    assert state is not None


# ── Mono-only / PCM16-only guards (LOW-R3) ────────────────────────────


def test_resample_rejects_non_mono():
    """channels != 1 must raise rather than silently corrupt interleaved audio."""
    pcm = b"\x00\x01" * 160
    with pytest.raises(ValueError, match="mono-only"):
        resample_audio(pcm, 8000, 16000, channels=2)


def test_resample_rejects_non_pcm16_width():
    """sample_width other than 2 must raise."""
    pcm = b"\x00\x01" * 160
    with pytest.raises(ValueError, match="PCM16-only"):
        resample_audio(pcm, 8000, 16000, sample_width=1)


def test_resample_mono_explicit_args_still_works():
    """Explicit mono/PCM16 args resample exactly as the defaults do."""
    pcm_8k = b"\x00\x01" * 160
    out, _ = resample_audio(pcm_8k, 8000, 16000, sample_width=2, channels=1)
    assert len(out) == 640


# --- StreamingResampler: non-integer ratios, chunk by chunk ---------------------


def _tone_pcm16(rate: int, seconds: float, freq: float = 440.0, amplitude: int = 8000) -> bytes:
    n = int(rate * seconds)
    samples = amplitude * np.sin(2.0 * np.pi * freq * np.arange(n) / rate)
    return samples.astype(np.int16).tobytes()


def _split(data: bytes, sizes) -> list:
    chunks, offset = [], 0
    for size in sizes:
        chunks.append(data[offset : offset + size])
        offset += size
    if offset < len(data):
        chunks.append(data[offset:])
    return chunks


@pytest.mark.parametrize("mode", ["linear", "bandlimited", "fir"])
@pytest.mark.parametrize(("source_rate", "target_rate"), [(44100, 8000), (44100, 16000), (24000, 8000), (16000, 8000)])
def test_streaming_resampler_chunking_does_not_change_the_result(mode, source_rate, target_rate):
    pcm = _tone_pcm16(source_rate, 0.5)

    whole = StreamingResampler(source_rate, target_rate, mode)
    expected = whole.process(pcm) + whole.flush()

    chunked = StreamingResampler(source_rate, target_rate, mode)
    # Uneven chunks, some of them odd-length in samples, one of them tiny.
    pieces = _split(pcm, [2 * 1000, 2 * 333, 2 * 7, 2 * 4096, 2 * 999, 2 * 1])
    produced = b"".join(chunked.process(piece) for piece in pieces) + chunked.flush()

    assert len(produced) == len(expected)
    got = np.frombuffer(produced, dtype=np.int16).astype(np.int32)
    want = np.frombuffer(expected, dtype=np.int16).astype(np.int32)
    # Only floating-point rounding of the carried position may differ.
    assert int(np.max(np.abs(got - want))) <= 1


@pytest.mark.parametrize(("source_rate", "target_rate"), [(44100, 8000), (44100, 16000), (22050, 8000)])
def test_streaming_resampler_output_length_follows_the_ratio_without_drift(source_rate, target_rate):
    seconds = 3.0
    pcm = _tone_pcm16(source_rate, seconds)
    resampler = StreamingResampler(source_rate, target_rate, "linear")
    produced = b"".join(resampler.process(piece) for piece in _split(pcm, [2 * 441] * 300))
    produced += resampler.flush()
    expected_samples = round(len(pcm) // 2 * target_rate / source_rate)
    assert abs(len(produced) // 2 - expected_samples) <= 1


def test_streaming_resampler_bandlimits_before_downsampling():
    # A 6 kHz tone lies above the 8 kHz target band: linear interpolation folds
    # it back as a 2 kHz alias, the filtered modes remove it.
    pcm = _tone_pcm16(44100, 1.0, freq=6000.0)
    linear = StreamingResampler(44100, 8000, "linear")
    filtered = StreamingResampler(44100, 8000, "bandlimited")
    out_linear = np.frombuffer(linear.process(pcm) + linear.flush(), dtype=np.int16).astype(np.float64)
    out_filtered = np.frombuffer(filtered.process(pcm) + filtered.flush(), dtype=np.int16).astype(np.float64)
    # Skip the filter's settling time.
    rms_linear = float(np.sqrt(np.mean(out_linear[800:] ** 2)))
    rms_filtered = float(np.sqrt(np.mean(out_filtered[800:] ** 2)))
    assert rms_linear > 2000.0
    assert rms_filtered < rms_linear / 20.0


def test_streaming_resampler_passes_equal_rates_through():
    pcm = _tone_pcm16(16000, 0.1)
    resampler = StreamingResampler(16000, 16000, "bandlimited")
    assert resampler.process(pcm) == pcm
    assert resampler.flush() == b""
    assert resampler.filtered is False


def test_streaming_resampler_first_output_is_the_first_input_sample():
    pcm = struct.pack("<8h", 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000)
    resampler = StreamingResampler(16000, 8000, "linear")
    out = np.frombuffer(resampler.process(pcm), dtype=np.int16)
    assert out[0] == 1000
    assert list(out) == [1000, 3000, 5000, 7000]


def test_streaming_resampler_rejects_bad_arguments():
    with pytest.raises(ValueError):
        StreamingResampler(0, 8000)
    with pytest.raises(ValueError):
        StreamingResampler(16000, 8000, mode="cubic")
