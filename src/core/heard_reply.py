"""What the caller heard of an interrupted pipeline reply.

A pipeline reply is synthesized ahead of playback: ElevenLabs returns audio far
faster than the telephone consumes it, so when the caller interrupts, the text
that reached the conversation history (the whole reply in serial mode, the
sentences queued so far in overlap mode) is longer than what was heard. The
next LLM turn then answers as if the caller had heard an offer they never got.

This module keeps, per reply, the text segments handed to the TTS adapter and
the audio each one produced. When the barge-in reports how much audio had
reached the transport, the heard part is the segments that played in full plus
a proportional prefix of the one that was cut, trimmed to a word boundary and
marked with an ellipsis. Segments still being synthesized are measured by the
speech rate of the completed ones (or a default), so a reply cut while its
audio was still arriving is estimated rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

ELLIPSIS = "\u2026"
# Conversational TTS runs at roughly 15 characters per second.
DEFAULT_MS_PER_CHAR = 65.0
_TRAILING_PUNCTUATION = " ,;:\u2014-"


def word_prefix(text: str, fraction: float) -> str:
    """The first ``fraction`` of ``text`` cut back to a word boundary; "" when no word fits."""
    text = text.strip()
    if not text or fraction <= 0:
        return ""
    if fraction >= 1:
        return text
    limit = int(len(text) * fraction)
    if limit <= 0:
        return ""
    cut = text.rfind(" ", 0, limit + 1)
    if cut <= 0:
        return ""
    return text[:cut].rstrip(_TRAILING_PUNCTUATION)


@dataclass
class ReplySegment:
    """One piece of text handed to the TTS adapter and the audio it produced so far."""

    text: str
    audio_ms: float = 0.0
    complete: bool = False


@dataclass
class SpokenReply:
    """Bookkeeping for one pipeline reply on one playback stream."""

    call_id: str
    stream_id: str
    bytes_per_ms: float = 8.0
    # Audio already sent to the transport but not yet heard when the caller spoke.
    lead_ms: float = 200.0
    segments: List[ReplySegment] = field(default_factory=list)
    # All of the reply's audio has been queued (nothing more was coming).
    completed: bool = False
    interrupted: bool = False
    played_ms: Optional[float] = None
    heard_text: Optional[str] = None
    # The assistant text currently in the conversation history for this reply, if any.
    persisted_text: Optional[str] = None
    # What sits in front of this reply in its history entry: the heard part of
    # the reply it continues (see Engine._join_continued_reply_history).
    prefix_text: str = ""
    # The speech that cut it off came to nothing and a continuation was asked for.
    continued: bool = False

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def open_segment(self, text: str) -> None:
        self.segments.append(ReplySegment(text=str(text or "")))

    def add_audio_ms(self, ms: float) -> None:
        if self.segments and ms > 0:
            self.segments[-1].audio_ms += float(ms)

    def add_audio_bytes(self, count: int) -> None:
        if count > 0:
            self.add_audio_ms(float(count) / max(1e-6, float(self.bytes_per_ms)))

    def close_segment(self) -> None:
        if self.segments:
            self.segments[-1].complete = True

    def _ms_per_char(self) -> float:
        done = [seg for seg in self.segments if seg.complete and seg.text.strip() and seg.audio_ms > 0]
        if not done:
            return DEFAULT_MS_PER_CHAR
        chars = sum(len(seg.text.strip()) for seg in done)
        return (sum(seg.audio_ms for seg in done) / chars) if chars else DEFAULT_MS_PER_CHAR

    def _segment_duration_ms(self, seg: ReplySegment) -> float:
        if seg.complete:
            return float(seg.audio_ms)
        # Still being synthesized: what is queued is a lower bound on its length.
        return max(float(seg.audio_ms), len(seg.text.strip()) * self._ms_per_char())

    def heard_text_at(self, played_ms: float) -> str:
        """The text the caller could hear once ``played_ms`` of audio had reached the transport."""
        budget = max(0.0, float(played_ms) - float(self.lead_ms))
        parts: List[str] = []
        cut = not self.completed
        for seg in self.segments:
            text = seg.text.strip()
            if not text:
                continue
            duration = self._segment_duration_ms(seg)
            if duration <= 0:
                cut = True
                break
            if budget + 1e-6 >= duration:
                parts.append(text)
                budget -= duration
                continue
            prefix = word_prefix(text, budget / duration)
            if prefix:
                parts.append(prefix)
            cut = True
            break
        heard = " ".join(parts).strip()
        if heard and cut and not heard.endswith(ELLIPSIS):
            heard += (" " + ELLIPSIS) if heard[-1] in ".!?" else ELLIPSIS
        return heard

    def mark_interrupted(self, played_ms: float) -> str:
        """Record the interruption and return the heard text."""
        self.interrupted = True
        self.played_ms = float(played_ms)
        self.heard_text = self.heard_text_at(played_ms)
        return self.heard_text
