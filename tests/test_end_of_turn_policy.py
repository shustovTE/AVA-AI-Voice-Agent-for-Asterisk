"""End-of-turn policy for pipeline STT results.

Regression cover for callers being answered mid-sentence: streaming STT returns
a result at every phrase boundary, and the superseded policy started a turn on
the first result that met a word or character threshold.
"""

import pytest

from src.engine import EndOfTurnPolicy


# Results the caller never got to finish, taken verbatim from a production log.
TRUNCATED_RESULTS = [
    "на паричном рынке хочу санузел поседить",
    "тоже перепланировку",
    "нет смотри я хочу",
]


def test_default_window():
    policy = EndOfTurnPolicy()

    assert policy.silence_ms == EndOfTurnPolicy.DEFAULT_SILENCE_MS
    assert policy.silence_sec == pytest.approx(0.7)
    assert policy.max_wait_ms == 0.0
    assert policy.flush_delay() == pytest.approx(0.7)


def test_missing_options_fall_back_to_defaults():
    for options in (None, [], "nonsense", 7):
        policy = EndOfTurnPolicy(options)

        assert policy.silence_ms == EndOfTurnPolicy.DEFAULT_SILENCE_MS
        assert policy.max_wait_ms == 0.0
        assert policy.legacy_options == ()
        assert policy.ignored_options == ()


def test_silence_window_is_configured_in_milliseconds():
    policy = EndOfTurnPolicy({"end_of_turn_silence_ms": 1200})

    assert policy.silence_ms == 1200
    assert policy.flush_delay() == pytest.approx(1.2)


def test_zero_window_answers_immediately():
    policy = EndOfTurnPolicy({"end_of_turn_silence_ms": 0})

    assert policy.silence_ms == 0.0
    assert policy.flush_delay() == 0.0


def test_negative_windows_are_clamped():
    policy = EndOfTurnPolicy(
        {"end_of_turn_silence_ms": -400, "end_of_turn_max_wait_ms": -1}
    )

    assert policy.silence_ms == 0.0
    assert policy.max_wait_ms == 0.0


@pytest.mark.parametrize("result", TRUNCATED_RESULTS)
def test_length_no_longer_decides_anything(result):
    """A three-word fragment waits exactly as long as a finished sentence."""
    policy = EndOfTurnPolicy({"end_of_turn_silence_ms": 800})

    assert policy.flush_delay() == pytest.approx(0.8)
    assert policy.ignored_options == ()
    # No API exists to shorten the window on the strength of the text.
    assert not hasattr(policy, "threshold_met")


def test_numeric_strings_are_accepted():
    policy = EndOfTurnPolicy(
        {"end_of_turn_silence_ms": "900", "end_of_turn_max_wait_ms": "5000"}
    )

    assert policy.silence_ms == pytest.approx(900.0)
    assert policy.max_wait_ms == pytest.approx(5000.0)


def test_unparsable_values_fall_back_without_raising():
    policy = EndOfTurnPolicy(
        {"end_of_turn_silence_ms": "soon", "end_of_turn_max_wait_ms": {}}
    )

    assert policy.silence_ms == EndOfTurnPolicy.DEFAULT_SILENCE_MS
    assert policy.max_wait_ms == 0.0


def test_booleans_are_not_treated_as_numbers():
    policy = EndOfTurnPolicy({"end_of_turn_silence_ms": True})

    assert policy.silence_ms == EndOfTurnPolicy.DEFAULT_SILENCE_MS


def test_max_wait_is_disabled_by_default():
    policy = EndOfTurnPolicy({"end_of_turn_silence_ms": 1000})

    assert policy.max_wait_ms == 0.0
    assert policy.flush_delay(elapsed=600.0) == pytest.approx(1.0)


def test_max_wait_caps_a_caller_who_never_pauses():
    policy = EndOfTurnPolicy(
        {"end_of_turn_silence_ms": 1000, "end_of_turn_max_wait_ms": 3000}
    )

    assert policy.flush_delay(elapsed=0.0) == pytest.approx(1.0)
    assert policy.flush_delay(elapsed=2.5) == pytest.approx(0.5)
    assert policy.flush_delay(elapsed=3.0) == 0.0
    assert policy.flush_delay(elapsed=9.9) == 0.0


def test_flush_delay_without_elapsed_ignores_the_cap():
    policy = EndOfTurnPolicy(
        {"end_of_turn_silence_ms": 1000, "end_of_turn_max_wait_ms": 3000}
    )

    assert policy.flush_delay() == pytest.approx(1.0)


# --- deployed configs written against the superseded options -----------------


def test_legacy_silence_seconds_are_converted():
    policy = EndOfTurnPolicy({"aggregation_silence_sec": 1.4})

    assert policy.silence_ms == pytest.approx(1400.0)
    assert policy.legacy_options == ("aggregation_silence_sec",)


def test_legacy_aggregation_timeout_becomes_the_window():
    """An operator who tuned the old timeout meant this window."""
    policy = EndOfTurnPolicy({"aggregation_timeout_sec": 0.8})

    assert policy.silence_ms == pytest.approx(800.0)
    assert policy.legacy_options == ("aggregation_timeout_sec",)


def test_milliseconds_win_over_every_legacy_key():
    policy = EndOfTurnPolicy(
        {
            "end_of_turn_silence_ms": 500,
            "aggregation_silence_sec": 1.4,
            "aggregation_timeout_sec": 0.8,
        }
    )

    assert policy.silence_ms == 500
    assert policy.legacy_options == ()


def test_legacy_max_wait_seconds_are_converted():
    policy = EndOfTurnPolicy({"aggregation_max_wait_sec": 4})

    assert policy.max_wait_ms == pytest.approx(4000.0)
    assert policy.legacy_options == ("aggregation_max_wait_sec",)


def test_retired_thresholds_are_reported_not_applied():
    """The reported production config, which answered every fragment at once."""
    policy = EndOfTurnPolicy(
        {
            "aggregation_timeout_sec": 0.8,
            "aggregation_min_words": 1,
            "aggregation_min_chars": 8,
        }
    )

    assert policy.silence_ms == pytest.approx(800.0)
    assert policy.ignored_options == ("aggregation_min_words", "aggregation_min_chars")


def test_retired_wait_for_silence_switch_is_reported():
    policy = EndOfTurnPolicy({"aggregation_wait_for_silence": False})

    assert policy.ignored_options == ("aggregation_wait_for_silence",)
    assert policy.silence_ms == EndOfTurnPolicy.DEFAULT_SILENCE_MS
