"""The engine cuts the caller's utterances for a recognizer that decodes whole phrases.

Silero VAD runs in the engine on the frames that reach the recognizer, so the
engine can say where an utterance begins and ends instead of a second VAD on
the server: from a pre-roll before Silero's start to its stop, as one unit.
"""

from src.core.utterances import SttUtterance, UtteranceCutter

RATE = 16000


def _frame(ms: int, value: int = 0x11) -> bytes:
    return bytes([value, value]) * int(RATE * ms / 1000)


def _cutter(**kwargs) -> UtteranceCutter:
    defaults = dict(sample_rate=RATE, preroll_ms=100, max_ms=2000, keep_ms=5000)
    defaults.update(kwargs)
    return UtteranceCutter(**defaults)


def test_an_utterance_runs_from_the_preroll_to_the_stop():
    cutter = _cutter()
    for _ in range(10):
        cutter.append(_frame(20, 0x01))  # 200 ms before the caller speaks
    cutter.speech_started()  # Silero confirmed speech: the onset is 100 ms back
    for _ in range(15):
        cutter.append(_frame(20, 0x22))  # 300 ms of speech and stop silence
    utterance = cutter.speech_stopped()

    assert isinstance(utterance, SttUtterance)
    assert utterance.duration_ms == 400  # 100 ms pre-roll + 300 ms
    assert utterance.pcm16[:2] == b"\x01\x01" and utterance.pcm16[-2:] == b"\x22\x22"
    assert utterance.signal_ms == 400
    assert utterance.reason == "stop" and utterance.utterance_id == "utt-1"
    assert utterance.ended_at >= utterance.started_at
    assert not cutter.open and cutter.speech_stopped() is None


def test_muted_frames_are_zeros_and_do_not_count_as_the_callers_audio():
    cutter = _cutter(preroll_ms=0)
    cutter.speech_started()
    cutter.append(_frame(100, 0x33), muted=True)  # the agent was audible
    cutter.append(_frame(100, 0x44))
    utterance = cutter.speech_stopped()

    assert utterance.duration_ms == 200
    assert set(utterance.pcm16[: RATE * 100 // 1000 * 2]) == {0}
    assert utterance.pcm16[-2:] == b"\x44\x44"
    assert utterance.signal_ms == 100


def test_a_long_utterance_is_split_at_a_quiet_chunk_in_its_second_half():
    cutter = _cutter(preroll_ms=0, max_ms=1000)
    cutter.speech_started()
    for i in range(40):  # 800 ms
        cutter.append(_frame(20, 0x55))
    cutter.mark_quiet()  # the newest chunk scored quiet at 800 ms
    for _ in range(10):  # 1000 ms reached
        cutter.append(_frame(20, 0x66))
    piece = cutter.split_overflow()

    assert piece is not None and piece.reason == "overflow"
    assert piece.duration_ms == 800  # cut at the quiet chunk, not at the cap
    assert cutter.open and cutter.open_ms == 200  # the rest goes on as the same utterance
    assert cutter.split_overflow() is None

    rest = cutter.speech_stopped()
    assert rest.duration_ms == 200 and rest.pcm16[:2] == b"\x66\x66"
    assert rest.utterance_id == "utt-2"


def test_a_long_utterance_without_a_recent_quiet_chunk_is_split_at_the_cap():
    cutter = _cutter(preroll_ms=0, max_ms=1000)
    cutter.speech_started()
    cutter.append(_frame(200))
    cutter.mark_quiet()  # too early: in the first half
    cutter.append(_frame(800))
    piece = cutter.split_overflow()

    assert piece is not None and piece.duration_ms == 1000
    assert cutter.open and cutter.open_ms == 0


def test_flush_returns_what_the_caller_was_saying_when_the_call_ended():
    cutter = _cutter(preroll_ms=0)
    assert cutter.flush() is None
    cutter.speech_started()
    cutter.append(_frame(150, 0x77))
    last = cutter.flush()

    assert last is not None and last.reason == "hangup" and last.duration_ms == 150
    assert not cutter.open


def test_old_audio_is_forgotten_but_never_the_open_utterance():
    cutter = _cutter(preroll_ms=0, max_ms=1000, keep_ms=0)  # keeps at least max + preroll + 1 s
    cutter.speech_started()
    for _ in range(150):  # 3 s: past the keep window
        cutter.append(_frame(20, 0x11))
    utterance = cutter.speech_stopped()

    assert utterance.duration_ms == 3000  # nothing of the open utterance was dropped
    for _ in range(200):
        cutter.append(_frame(20))
    assert cutter.read(0, RATE) == bytes(RATE * 2)  # gone audio reads as silence


def test_the_preroll_never_reaches_before_the_first_frame():
    cutter = _cutter(preroll_ms=500)
    cutter.append(_frame(40, 0x11))
    cutter.speech_started()
    cutter.append(_frame(60, 0x22))
    utterance = cutter.speech_stopped()

    assert utterance.duration_ms == 100
