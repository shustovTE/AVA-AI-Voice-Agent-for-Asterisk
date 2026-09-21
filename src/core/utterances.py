"""Whole utterances for a recognizer that decodes phrases, cut by the engine's own VAD.

A recognizer that decodes one phrase at a time (GigaAM v3 and NeMo through
onnx-asr, Sherpa offline, the Whisper family) needs someone to say where a
phrase begins and ends. Left to the server, that someone is a second voice
activity detector fed with a continuous stream, which closes a phrase only
after its own silence gate and knows nothing about the engine's decision that
the caller's turn is over. With Silero VAD running in the engine on the very
same frames, the engine can cut the utterance itself: the audio from a little
before Silero's ``start`` to its ``stop`` is one utterance, sent to the
recognizer as one unit the moment the caller stops, and the recognizer only
decodes it.

:class:`UtteranceCutter` keeps the caller's recent audio at the recognizer's
rate and hands out one utterance per Silero stop (with a pre-roll for the
onset Silero needed to make up its mind), or a piece of a very long one so a
model trained on short clips is never given a minute of speech at once.
:class:`SttUtterance` is the item that carries such an utterance through the
pipeline's STT queue next to the raw frames of the streaming path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SttUtterance:
    """One caller utterance for the recognizer, cut by the engine's VAD."""

    pcm16: bytes
    sample_rate: int
    utterance_id: str
    # Monotonic times of the utterance's first and last sample.
    started_at: float
    ended_at: float
    # Why it was cut: "stop" (Silero reported the caller quiet), "overflow"
    # (a long utterance split at its length cap) or "hangup" (the call ended).
    reason: str = "stop"
    # How much of it is real caller audio: frames muted while the agent was
    # audible are zeros and do not count.
    signal_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        return len(self.pcm16) / 2.0 / max(1, int(self.sample_rate)) * 1000.0


@dataclass
class _Chunk:
    start: int
    data: bytes
    muted: bool

    @property
    def samples(self) -> int:
        return len(self.data) // 2

    @property
    def end(self) -> int:
        return self.start + self.samples


@dataclass
class _OpenSegment:
    start: int
    # Absolute sample position after the newest chunk that scored quiet, for a
    # split that does not land in the middle of a word.
    quiet_at: Optional[int] = None
    started_at: float = field(default_factory=time.monotonic)


class UtteranceCutter:
    """The caller's recent audio, cut into utterances at Silero's start and stop.

    Audio is appended frame by frame as it reaches the engine, muted frames as
    zeros so the timeline stays continuous. ``speech_started`` marks where an
    utterance begins (``preroll_ms`` before the frame that opened it, since
    Silero needs ``start_ms`` of speech before it reports a start);
    ``speech_stopped`` returns everything from there to the frame that closed
    it, which already includes Silero's stop silence. An utterance that grows
    past ``max_ms`` is split by ``split_overflow`` at the newest quiet chunk in
    its second half, or at its end when none was seen.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        preroll_ms: float = 300.0,
        max_ms: float = 20000.0,
        keep_ms: float = 30000.0,
    ) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.preroll_samples = int(max(0.0, float(preroll_ms)) * self.sample_rate / 1000.0)
        self.max_samples = int(max(1000.0, float(max_ms)) * self.sample_rate / 1000.0)
        self._keep_samples = max(
            int(max(0.0, float(keep_ms)) * self.sample_rate / 1000.0),
            self.max_samples + self.preroll_samples + self.sample_rate,
        )
        self._chunks: List[_Chunk] = []
        # Absolute sample position of the next sample to be appended.
        self.position = 0
        self._open: Optional[_OpenSegment] = None
        self.utterances = 0

    # ── feeding ──────────────────────────────────────────────────────────────
    def append(self, pcm16: bytes, *, muted: bool = False) -> None:
        """Add one frame of PCM16 at the cutter's rate; a muted frame is stored as zeros."""
        if not pcm16:
            return
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        samples = len(pcm16) // 2
        if samples <= 0:
            return
        data = bytes(samples * 2) if muted else bytes(pcm16)
        self._chunks.append(_Chunk(start=self.position, data=data, muted=bool(muted)))
        self.position += samples
        self._trim()

    def mark_quiet(self) -> None:
        """The newest chunk scored quiet: a candidate split point for a long utterance."""
        if self._open is not None:
            self._open.quiet_at = self.position

    def _trim(self) -> None:
        floor = self.position - self._keep_samples
        if self._open is not None:
            floor = min(floor, self._open.start)
        while len(self._chunks) > 1 and self._chunks[0].end <= floor:
            self._chunks.pop(0)

    # ── segment lifecycle ────────────────────────────────────────────────────
    @property
    def open(self) -> bool:
        return self._open is not None

    @property
    def open_ms(self) -> float:
        if self._open is None:
            return 0.0
        return (self.position - self._open.start) / self.sample_rate * 1000.0

    @property
    def open_signal_ms(self) -> float:
        if self._open is None:
            return 0.0
        return self._signal_samples_between(self._open.start, self.position) / self.sample_rate * 1000.0

    @property
    def open_started_at(self) -> Optional[float]:
        return self._open.started_at if self._open is not None else None

    def speech_started(self) -> None:
        """Silero reported speech: the utterance begins ``preroll_ms`` before now."""
        if self._open is not None:
            return
        oldest = self._chunks[0].start if self._chunks else self.position
        start = max(oldest, self.position - self.preroll_samples)
        self._open = _OpenSegment(start=start)

    def speech_stopped(self, *, reason: str = "stop") -> Optional[SttUtterance]:
        """Silero reported the caller quiet: return the open utterance and close it."""
        if self._open is None:
            return None
        return self._cut(self.position, reason=reason, close=True)

    def flush(self, *, reason: str = "hangup") -> Optional[SttUtterance]:
        """Return the open utterance as it stands (the call is ending) and close it."""
        return self.speech_stopped(reason=reason)

    def split_overflow(self) -> Optional[SttUtterance]:
        """When the open utterance has reached its cap, cut its first part and keep going."""
        if self._open is None or self.position - self._open.start < self.max_samples:
            return None
        quiet_at = self._open.quiet_at
        half = self._open.start + self.max_samples // 2
        cut_at = quiet_at if quiet_at is not None and quiet_at >= half else self.position
        return self._cut(cut_at, reason="overflow", close=False)

    # ── internals ────────────────────────────────────────────────────────────
    def _cut(self, end: int, *, reason: str, close: bool) -> Optional[SttUtterance]:
        assert self._open is not None
        start = self._open.start
        if end <= start:
            if close:
                self._open = None
            return None
        pcm = self.read(start, end)
        signal = self._signal_samples_between(start, end)
        now = time.monotonic()
        self.utterances += 1
        utterance = SttUtterance(
            pcm16=pcm,
            sample_rate=self.sample_rate,
            utterance_id=f"utt-{self.utterances}",
            started_at=now - (self.position - start) / self.sample_rate,
            ended_at=now - (self.position - end) / self.sample_rate,
            reason=reason,
            signal_ms=signal / self.sample_rate * 1000.0,
        )
        if close:
            self._open = None
        else:
            self._open = _OpenSegment(start=end)
        return utterance

    def read(self, start: int, end: int) -> bytes:
        """The buffered audio between two absolute sample positions (zeros where it is gone)."""
        if end <= start:
            return b""
        out = bytearray()
        pos = start
        for chunk in self._chunks:
            if chunk.end <= pos:
                continue
            if chunk.start >= end:
                break
            if chunk.start > pos:
                out.extend(bytes((chunk.start - pos) * 2))
                pos = chunk.start
            take_from = pos - chunk.start
            take_to = min(chunk.samples, end - chunk.start)
            out.extend(chunk.data[take_from * 2 : take_to * 2])
            pos = chunk.start + take_to
            if pos >= end:
                break
        if pos < end:
            out.extend(bytes((end - pos) * 2))
        return bytes(out)

    def _signal_samples_between(self, start: int, end: int) -> int:
        """Samples between two positions that came from unmuted frames."""
        total = 0
        for chunk in self._chunks:
            if chunk.end <= start:
                continue
            if chunk.start >= end:
                break
            if chunk.muted:
                continue
            total += min(chunk.end, end) - max(chunk.start, start)
        return max(0, total)
