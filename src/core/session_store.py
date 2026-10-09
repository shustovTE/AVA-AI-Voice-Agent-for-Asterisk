"""
SessionStore - Centralized, atomic state management for call sessions.

This replaces the dict soup (active_calls, caller_channels, active_playbacks)
with a single, thread-safe store that enforces invariants.
"""

import asyncio
import inspect
import time
from typing import Optional, Dict, Set, List
import structlog

from src.core.models import CallSession, PlaybackRef, ProviderSession

logger = structlog.get_logger(__name__)


class SessionStore:
    """
    Thread-safe store for call sessions and playback references.
    
    Enforces key invariants:
    - Canonical call_id == caller_channel_id
    - A call has two channel entries (caller/local), both share call_id
    - Gating is token/refcount-based per call
    - All operations are atomic
    """
    
    def __init__(self):
        # Core session storage
        self._sessions_by_call_id: Dict[str, CallSession] = {}
        self._sessions_by_channel_id: Dict[str, CallSession] = {}
        self._playbacks: Dict[str, PlaybackRef] = {}
        self._provider_sessions: Dict[str, ProviderSession] = {}
        # Calls removed by call cleanup, call_id -> epoch seconds. A late writer that
        # still holds the removed CallSession object (a pipeline task finishing its
        # request, a coordinator timer, an AudioSocket bind that raced the hangup)
        # must not put it back: nothing would ever remove it again, because the
        # channel behind it is gone and Asterisk sends no further events for it.
        self._tombstones: Dict[str, float] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info("SessionStore initialized")
    
    TOMBSTONE_TTL_SECONDS = 3600.0
    TOMBSTONE_MAX_ENTRIES = 2000

    async def upsert_call(self, session: CallSession, *, allow_resurrect: bool = False) -> bool:
        """Add or update a call session atomically.

        Returns False when the write was refused because the call was already
        removed by cleanup (see ``_tombstones``); ``allow_resurrect=True`` overrides
        that for callers that deliberately rebuild a session.
        """
        async with self._lock:
            call_id = session.call_id
            if (
                not allow_resurrect
                and call_id in self._tombstones
                and call_id not in self._sessions_by_call_id
            ):
                logger.warning(
                    "Refusing to resurrect a cleaned-up call session",
                    call_id=call_id,
                    status=getattr(session, "status", None),
                    conversation_state=getattr(session, "conversation_state", None),
                    writer=self._writer_location(),
                )
                return False
            # Store by call_id (canonical)
            self._sessions_by_call_id[session.call_id] = session
            
            # Store by caller_channel_id
            self._sessions_by_channel_id[session.caller_channel_id] = session
            
            # Store by local_channel_id if present
            if session.local_channel_id:
                self._sessions_by_channel_id[session.local_channel_id] = session
            
            # Store by external_media_id if present
            if session.external_media_id:
                self._sessions_by_channel_id[session.external_media_id] = session
            
            # Store by audiosocket_channel_id if present
            if session.audiosocket_channel_id:
                self._sessions_by_channel_id[session.audiosocket_channel_id] = session
            
            logger.debug("Call session upserted",
                        call_id=session.call_id,
                        caller_channel_id=session.caller_channel_id,
                        local_channel_id=session.local_channel_id)
            return True

    @staticmethod
    def _writer_location() -> str:
        """Name the code that asked for the write: the first frame outside this
        module and outside the engine's ``_save_session`` wrapper."""
        frame = inspect.currentframe()
        try:
            f = frame.f_back if frame is not None else None
            hops = 0
            while f is not None and hops < 12:
                name = f.f_code.co_name
                if not f.f_code.co_filename.endswith("session_store.py") and name not in (
                    "_save_session",
                    "upsert_call",
                ):
                    return f"{name}:{f.f_lineno}"
                f = f.f_back
                hops += 1
        except Exception:
            pass
        finally:
            del frame
        return "unknown"

    def is_tombstoned(self, call_id: str) -> bool:
        """True when call cleanup already removed this call's session."""
        return call_id in self._tombstones

    async def forget_tombstone(self, call_id: str) -> None:
        async with self._lock:
            self._tombstones.pop(call_id, None)

    def _prune_tombstones_locked(self, now: float) -> None:
        cutoff = now - self.TOMBSTONE_TTL_SECONDS
        stale = [cid for cid, ts in self._tombstones.items() if ts < cutoff]
        for cid in stale:
            self._tombstones.pop(cid, None)
        overflow = len(self._tombstones) - self.TOMBSTONE_MAX_ENTRIES
        if overflow > 0:
            for cid, _ts in sorted(self._tombstones.items(), key=lambda kv: kv[1])[:overflow]:
                self._tombstones.pop(cid, None)
    
    async def get_by_call_id(self, call_id: str) -> Optional[CallSession]:
        """Get session by canonical call_id."""
        async with self._lock:
            return self._sessions_by_call_id.get(call_id)

    async def append_tool_call_if_active(self, call_id: str, record: dict) -> bool:
        """Append history only while the call is registered, under one lock."""
        async with self._lock:
            session = self._sessions_by_call_id.get(call_id)
            if session is None:
                return False
            if session.tool_calls is None:
                session.tool_calls = []
            session.tool_calls.append(record)
            return True
    
    async def get_by_channel_id(self, channel_id: str) -> Optional[CallSession]:
        """Get session by any channel_id (caller, local, external_media)."""
        async with self._lock:
            return self._sessions_by_channel_id.get(channel_id)

    async def has_active_sessions_for_provider(self, provider_key: str) -> bool:
        """Return whether any active call is currently using provider_key."""
        async with self._lock:
            for session in self._sessions_by_call_id.values():
                if getattr(session, "provider_name", None) == provider_key:
                    return True
            return False
    
    async def remove_call(self, call_id: str, *, tombstone: bool = False) -> Optional[CallSession]:
        """Remove a call session and all its channel mappings.

        ``tombstone=True`` (call cleanup) also records the removal so that a late
        ``upsert_call`` with the stale object is refused instead of resurrecting it.
        """
        async with self._lock:
            if tombstone:
                now = time.time()
                self._tombstones[call_id] = now
                self._prune_tombstones_locked(now)
            session = self._sessions_by_call_id.pop(call_id, None)
            if not session:
                return None
            
            # Remove all channel mappings
            self._sessions_by_channel_id.pop(session.caller_channel_id, None)
            if session.local_channel_id:
                self._sessions_by_channel_id.pop(session.local_channel_id, None)
            if session.external_media_id:
                self._sessions_by_channel_id.pop(session.external_media_id, None)
            if session.audiosocket_channel_id:
                self._sessions_by_channel_id.pop(session.audiosocket_channel_id, None)
            
            logger.debug("Call session removed",
                        call_id=call_id,
                        caller_channel_id=session.caller_channel_id)
            
            return session
    
    async def set_gating_token(self, call_id: str, playback_id: str) -> bool:
        """Add a TTS gating token for a call."""
        async with self._lock:
            session = self._sessions_by_call_id.get(call_id)
            if not session:
                logger.warning("Cannot set gating token - call not found", 
                             call_id=call_id, playback_id=playback_id)
                return False

            already_present = playback_id in session.tts_tokens
            prior_count = int(getattr(session, "tts_active_count", 0) or 0)

            session.tts_tokens.add(playback_id)
            # Invariant: active_count must match the token set size (idempotent under duplicate events).
            session.tts_active_count = len(session.tts_tokens)

            # Only apply "start" side effects on the 0->1 transition.
            if prior_count == 0 and session.tts_active_count > 0:
                session.tts_playing = True
                session.audio_capture_enabled = False
                # Record TTS start time for barge-in protection window
                session.tts_started_ts = time.time()

                # Update VAD state
                if session.vad_state:
                    session.vad_state["tts_playing"] = True
                    # Reset VAD buffers to prevent TTS bleed-through
                    session.vad_state["webrtc_speech_frames"] = 0
                    session.vad_state["webrtc_silence_frames"] = 0
                    session.vad_state["webrtc_last_decision"] = False
                    # ARCHITECT FIX: Reset both audio_buffer and frame_buffer
                    if "audio_buffer" in session.vad_state:
                        session.vad_state["audio_buffer"] = b""
                    if "frame_buffer" in session.vad_state:
                        session.vad_state["frame_buffer"] = b""
            else:
                # Ensure we remain gated when tokens exist (even if pre-state drifted).
                if session.tts_active_count > 0:
                    session.tts_playing = True
                    session.audio_capture_enabled = False
            
            logger.info("🔇 TTS GATING - Token added",
                       call_id=call_id,
                       playback_id=playback_id,
                       active_count=session.tts_active_count,
                       audio_capture_enabled=session.audio_capture_enabled,
                       tts_playing=session.tts_playing,
                       already_present=already_present)
            
            return True
    
    async def clear_gating_token(self, call_id: str, playback_id: str) -> bool:
        """Remove a TTS gating token for a call."""
        async with self._lock:
            session = self._sessions_by_call_id.get(call_id)
            if not session:
                logger.warning("Cannot clear gating token - call not found",
                             call_id=call_id, playback_id=playback_id)
                return False

            was_present = playback_id in session.tts_tokens
            prior_count = int(getattr(session, "tts_active_count", 0) or 0)

            session.tts_tokens.discard(playback_id)
            # Invariant: active_count must match the token set size (idempotent under duplicate events).
            session.tts_active_count = len(session.tts_tokens)

            # Only apply "end" side effects on the 1->0 transition.
            if prior_count > 0 and session.tts_active_count == 0:
                session.tts_playing = False
                session.audio_capture_enabled = True
                # Record TTS end time so the engine can guard a short post-TTS window
                try:
                    session.tts_ended_ts = time.time()
                except Exception:
                    session.tts_ended_ts = 0.0

                # Update VAD state
                if session.vad_state:
                    session.vad_state["tts_playing"] = False
                    # ARCHITECT FIX: Reset both audio_buffer and frame_buffer
                    if "audio_buffer" in session.vad_state:
                        session.vad_state["audio_buffer"] = b""
                    if "frame_buffer" in session.vad_state:
                        session.vad_state["frame_buffer"] = b""
            else:
                # Ensure we remain gated if any tokens remain (even if pre-state drifted).
                if session.tts_active_count > 0:
                    session.tts_playing = True
                    session.audio_capture_enabled = False
            
            logger.info("🔊 TTS GATING - Token removed",
                       call_id=call_id,
                       playback_id=playback_id,
                       active_count=session.tts_active_count,
                       audio_capture_enabled=session.audio_capture_enabled,
                       tts_playing=session.tts_playing,
                       was_present=was_present)
            
            return True
    
    async def add_playback(self, playback_ref: PlaybackRef) -> None:
        """Add a playback reference."""
        async with self._lock:
            self._playbacks[playback_ref.playback_id] = playback_ref
            logger.debug("Playback reference added",
                        playback_id=playback_ref.playback_id,
                        call_id=playback_ref.call_id)
    
    async def pop_playback(self, playback_id: str) -> Optional[PlaybackRef]:
        """Remove and return a playback reference."""
        async with self._lock:
            playback_ref = self._playbacks.pop(playback_id, None)
            if playback_ref:
                logger.debug("Playback reference removed",
                           playback_id=playback_id,
                           call_id=playback_ref.call_id)
            return playback_ref
    
    async def get_playback(self, playback_id: str) -> Optional[PlaybackRef]:
        """Get a playback reference without removing it."""
        async with self._lock:
            return self._playbacks.get(playback_id)

    async def list_playbacks_for_call(self, call_id: str) -> List[str]:
        """List playback IDs associated with a given call_id."""
        async with self._lock:
            return [pid for pid, pref in self._playbacks.items() if pref.call_id == call_id]
    
    async def list_active_calls(self) -> List[str]:
        """Get list of active call IDs."""
        async with self._lock:
            return list(self._sessions_by_call_id.keys())
    
    async def get_all_sessions(self) -> List[CallSession]:
        """Get all active sessions."""
        async with self._lock:
            return list(self._sessions_by_call_id.values())

    async def count_active_outbound_calls(
        self,
        campaign_id: Optional[str] = None,
        *,
        excluding_attempt_ids: Optional[Set[str]] = None,
    ) -> int:
        """Count active outbound calls, optionally excluding attempts counted elsewhere."""
        excluded = {str(value) for value in (excluding_attempt_ids or set()) if str(value)}
        async with self._lock:
            count = 0
            for session in self._sessions_by_call_id.values():
                if not getattr(session, "is_outbound", False):
                    continue
                if campaign_id and getattr(session, "outbound_campaign_id", None) != campaign_id:
                    continue
                attempt_id = str(getattr(session, "outbound_attempt_id", None) or "").strip()
                if attempt_id and attempt_id in excluded:
                    continue
                count += 1
            return count
    
    async def get_session_stats(self) -> Dict[str, any]:
        """Get statistics about active sessions including per-call details."""
        async with self._lock:
            # Build list of active call details for Admin UI topology
            active_sessions = []
            now = time.time()
            for call_id, session in self._sessions_by_call_id.items():
                created_at = float(getattr(session, "created_at", 0.0) or 0.0)
                active_sessions.append({
                    "call_id": call_id,
                    "caller_channel_id": session.caller_channel_id,
                    "provider": session.provider_name,
                    "pipeline": session.pipeline_name,
                    "context": session.context_name,
                    "status": session.status,
                    "conversation_state": session.conversation_state,
                    "is_outbound": bool(getattr(session, "is_outbound", False)),
                    "cleanup_in_progress": bool(getattr(session, "cleanup_in_progress", False)),
                    "created_at": created_at,
                    "age_seconds": round(max(0.0, now - created_at), 1) if created_at else None,
                })
            
            return {
                "active_calls": len(self._sessions_by_call_id),
                "active_playbacks": len(self._playbacks),
                "provider_sessions": len(self._provider_sessions),
                "sessions": active_sessions,
            }
    
    async def cleanup_expired_sessions(self, max_age_seconds: float = 3600) -> int:
        """Clean up sessions older than max_age_seconds."""
        # First pass: identify expired calls while holding lock
        async with self._lock:
            current_time = time.time()
            expired_calls = []
            
            for call_id, session in self._sessions_by_call_id.items():
                if current_time - session.created_at > max_age_seconds:
                    expired_calls.append(call_id)
        
        # Second pass: remove expired calls (each remove_call acquires its own lock)
        for call_id in expired_calls:
            await self.remove_call(call_id)
        
        if expired_calls:
            logger.info("Cleaned up expired sessions",
                       expired_count=len(expired_calls))
        
        return len(expired_calls)
