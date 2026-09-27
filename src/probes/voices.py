"""Register a reference voice on a self-hosted speech endpoint.

vLLM-Omni (Fish Speech S2-Pro and the other cloning models it serves) keeps a
registry of reference voices: ``POST /v1/audio/voices`` takes the sample, its
transcript and a name, and a request may then say ``voice: <name>`` instead
of carrying the sample. The sample must be uploaded, the server accepts no
path, so the engine, which shares a host with the server, reads it from a
directory mounted into its container (``tts_voices_dir``, ``/voices`` by
default) and uploads it. The Admin UI asks the engine to do this; the file
name is a bare name inside that directory, never a path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import aiohttp

from ..config.provider_instances import resolve_secret_value
from .providers import openai_probe_base_url, substitute_env_vars, url_host

logger = logging.getLogger(__name__)

MAX_SAMPLE_BYTES = 10 * 1024 * 1024
DEFAULT_VOICES_DIR = "/voices"
DEFAULT_CONSENT = "ava-admin"
UPLOAD_TIMEOUT_SEC = 120.0

_AUDIO_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
}


class VoiceRegistrationError(ValueError):
    """A request the engine refuses before talking to the server."""


def resolve_sample_path(voices_dir: str, file_name: str) -> Path:
    """The sample file inside the voices directory, or a refusal that names why."""
    name = str(file_name or "").strip()
    if not name:
        raise VoiceRegistrationError("File name is required")
    if name in (".", "..") or any(sep in name for sep in ("/", "\\", "\x00")):
        raise VoiceRegistrationError("File name must be a bare name inside the voices directory, without path separators")
    root = Path(str(voices_dir or DEFAULT_VOICES_DIR))
    if not root.is_dir():
        raise VoiceRegistrationError(
            f"Voices directory {root} is not available in the ai_engine container; mount it there "
            "(the same directory the speech server reads) or set tts_voices_dir on the provider"
        )
    path = root / name
    if not path.is_file():
        raise VoiceRegistrationError(f"File {name} not found in {root}")
    if path.suffix.lower() not in _AUDIO_TYPES:
        raise VoiceRegistrationError(
            f"Unsupported audio format {path.suffix or '(none)'}; use one of "
            + ", ".join(sorted(_AUDIO_TYPES))
        )
    size = path.stat().st_size
    if size > MAX_SAMPLE_BYTES:
        raise VoiceRegistrationError(f"File {name} is {size / 1048576:.1f} MB; the server accepts at most 10 MB")
    return path


def validate_voice_name(name: str) -> str:
    """The same rule the server applies: non-empty, no path separators or NUL."""
    trimmed = str(name or "").strip()
    if not trimmed or trimmed in (".", "..") or any(c in trimmed for c in "/\\\x00"):
        raise VoiceRegistrationError("Voice name must be non-empty, without path separators")
    return trimmed


async def _post_form(url: str, form: aiohttp.FormData, headers: Dict[str, str], timeout: float) -> tuple[int, str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, data=form, headers=headers) as resp:
            return resp.status, await resp.text()


def _server_error_text(status: int, text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    message: Any = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
        message = message or payload.get("detail") or payload.get("message")
    if isinstance(message, (dict, list)):
        message = json.dumps(message)
    message = str(message or text or "").strip().replace("\n", " ")
    return f"HTTP {status}: {message[:300]}" if message else f"HTTP {status}"


async def register_voice(
    provider_key: str,
    block: Mapping[str, Any],
    *,
    file: str,
    name: Optional[str] = None,
    ref_text: str,
    consent: Optional[str] = None,
    env: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Upload the sample *file* of the provider's voices directory as voice *name*.

    Returns ``{"success": bool, "message": str}`` and, on success, ``voice``
    with the registered name, which the provider block can then carry as
    ``voice``. The target host comes from the saved block, never from the
    request.
    """
    environment: Dict[str, str] = dict(os.environ)
    if env:
        environment.update({str(k): str(v) for k, v in env.items() if v is not None and str(v) != ""})
    cfg = substitute_env_vars(dict(block or {}), environment)

    if str(cfg.get("type") or "").strip().lower() != "openai":
        return {
            "success": False,
            "message": f"Provider '{provider_key}' is not an OpenAI-compatible speech block (type: openai)",
        }
    tts_base_url = str(cfg.get("tts_base_url") or "").strip()
    if not tts_base_url:
        return {"success": False, "message": f"Provider '{provider_key}' has no tts_base_url"}
    base_url = openai_probe_base_url({"tts_base_url": tts_base_url}).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        return {"success": False, "message": f"tts_base_url of '{provider_key}' is not an http(s) URL"}

    try:
        path = resolve_sample_path(str(cfg.get("tts_voices_dir") or DEFAULT_VOICES_DIR), file)
        voice_name = validate_voice_name(name or path.stem)
    except VoiceRegistrationError as exc:
        return {"success": False, "message": str(exc)}
    transcript = str(ref_text or "").strip()
    if not transcript:
        return {
            "success": False,
            "message": "Transcript (ref_text) is required: without it the server stores the sample but does not clone from it",
        }
    consent_id = str(consent or "").strip() or DEFAULT_CONSENT

    prefix = provider_key.rsplit("_", 1)[0] if "_" in provider_key else provider_key
    try:
        api_key = resolve_secret_value(
            cfg,
            file_field="api_key_file",
            env_field="api_key_env",
            inline_field="api_key",
            legacy_env_names=(f"{prefix.upper()}_API_KEY",),
        )
    except Exception:
        logger.warning("Voice registration could not resolve the provider API key", exc_info=True)
        api_key = str(cfg.get("api_key") or "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    size = path.stat().st_size
    form = aiohttp.FormData()
    form.add_field(
        "audio_sample",
        path.read_bytes(),
        filename=path.name,
        content_type=_AUDIO_TYPES[path.suffix.lower()],
    )
    form.add_field("name", voice_name)
    form.add_field("ref_text", transcript)
    form.add_field("consent", consent_id)

    url = f"{base_url}/audio/voices"
    host = url_host(url) or base_url
    try:
        status, text = await _post_form(url, form, headers, UPLOAD_TIMEOUT_SEC)
    except aiohttp.ClientConnectorError as exc:
        return {"success": False, "message": f"Cannot connect to the speech server at {base_url}: {exc}"}
    except asyncio.TimeoutError:
        return {"success": False, "message": f"The speech server at {base_url} did not answer within {UPLOAD_TIMEOUT_SEC:.0f} s"}
    except aiohttp.ClientError as exc:
        return {"success": False, "message": f"Upload to {base_url} failed: {exc}"}

    if status != 200:
        return {"success": False, "message": f"The speech server refused the voice ({_server_error_text(status, text)})"}
    try:
        payload = json.loads(text)
    except Exception:
        payload = {}
    voice_info = payload.get("voice") if isinstance(payload, dict) else None
    registered = str((voice_info or {}).get("name") or voice_name) if isinstance(voice_info, dict) else voice_name
    return {
        "success": True,
        "voice": registered,
        "message": (
            f"Voice '{registered}' registered at {host} from {path.name} "
            f"({size / 1024:.0f} KB, transcript {len(transcript)} chars). "
            f"Set voice: {registered} on the provider to use it."
        ),
    }
