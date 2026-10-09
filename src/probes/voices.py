"""Register and list reference voices on a self-hosted speech endpoint.

vLLM-Omni (Fish Speech S2-Pro and the other cloning models it serves) keeps a
registry of reference voices: ``POST /v1/audio/voices`` takes the sample, its
transcript and a name, and a request may then say ``voice: <name>`` instead
of carrying the sample; ``GET /v1/audio/voices`` lists what is registered.
The server accepts only an upload, no path, so the sample's bytes travel
from wherever they are to the engine, which shares a host and a network
with the server: from the operator's browser through the Admin UI (the
usual way), from a file the Admin UI staged into the engine container
through the Docker socket, or, for scripts, from a file in a directory
mounted into the engine (``tts_voices_dir``). The target host always comes
from the saved provider block, never from the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import aiohttp

from ..config.provider_instances import resolve_secret_value
from .providers import openai_probe_base_url, substitute_env_vars, url_host

logger = logging.getLogger(__name__)

MAX_SAMPLE_BYTES = 10 * 1024 * 1024
DEFAULT_VOICES_DIR = "/voices"
DEFAULT_CONSENT = "ava-admin"
UPLOAD_TIMEOUT_SEC = 120.0
LIST_TIMEOUT_SEC = 15.0
# Where the Admin UI puts a sample it cannot send over the network (the
# engine's health server is loopback-only) and hands to the engine through
# the Docker socket; the engine deletes each file after reading it.
STAGING_DIR = Path(os.environ.get("AVA_VOICE_STAGING_DIR") or "/tmp/ava-voice-uploads")

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


def _bare_name(value: str, what: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise VoiceRegistrationError(f"{what} is required")
    if name in (".", "..") or any(sep in name for sep in ("/", "\\", "\x00")):
        raise VoiceRegistrationError(f"{what} must be a bare file name, without path separators")
    return name


def validate_voice_name(name: str) -> str:
    """The same rule the server applies: non-empty, no path separators or NUL."""
    trimmed = str(name or "").strip()
    if not trimmed or trimmed in (".", "..") or any(c in trimmed for c in "/\\\x00"):
        raise VoiceRegistrationError("Voice name must be non-empty, without path separators")
    return trimmed


def _content_type_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    content_type = _AUDIO_TYPES.get(suffix)
    if content_type is None:
        raise VoiceRegistrationError(
            f"Unsupported audio format {suffix or '(none)'}; use one of " + ", ".join(sorted(_AUDIO_TYPES))
        )
    return content_type


def resolve_sample_path(voices_dir: str, file_name: str) -> Path:
    """The sample file inside the voices directory, or a refusal that names why."""
    name = _bare_name(file_name, "File name")
    root = Path(str(voices_dir or DEFAULT_VOICES_DIR))
    if not root.is_dir():
        raise VoiceRegistrationError(
            f"Voices directory {root} is not available in the ai_engine container; mount it there "
            "or upload the sample from the browser"
        )
    path = root / name
    if not path.is_file():
        raise VoiceRegistrationError(f"File {name} not found in {root}")
    _content_type_for(path.name)
    size = path.stat().st_size
    if size > MAX_SAMPLE_BYTES:
        raise VoiceRegistrationError(f"File {name} is {size / 1048576:.1f} MB; the server accepts at most 10 MB")
    return path


def take_staged_sample(staged_file: str) -> Tuple[bytes, str]:
    """Read, then delete, a sample the Admin UI staged through the Docker socket."""
    name = _bare_name(staged_file, "Staged file name")
    path = STAGING_DIR / name
    if not path.is_file():
        raise VoiceRegistrationError(f"Staged sample {name} not found in {STAGING_DIR}")
    try:
        data = path.read_bytes()
    finally:
        try:
            path.unlink()
        except OSError:
            logger.debug("Could not remove staged sample %s", path, exc_info=True)
    return data, name


def _resolve_endpoint(provider_key: str, block: Mapping[str, Any], env: Optional[Mapping[str, Optional[str]]]):
    """``(base_url, api_key, None)`` for a usable block, else ``(None, None, failure)``."""
    environment: Dict[str, str] = dict(os.environ)
    if env:
        environment.update({str(k): str(v) for k, v in env.items() if v is not None and str(v) != ""})
    cfg = substitute_env_vars(dict(block or {}), environment)

    if str(cfg.get("type") or "").strip().lower() != "openai":
        return None, None, {
            "success": False,
            "message": f"Provider '{provider_key}' is not an OpenAI-compatible speech block (type: openai)",
        }
    tts_base_url = str(cfg.get("tts_base_url") or "").strip()
    if not tts_base_url:
        return None, None, {"success": False, "message": f"Provider '{provider_key}' has no tts_base_url"}
    base_url = openai_probe_base_url({"tts_base_url": tts_base_url}).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        return None, None, {"success": False, "message": f"tts_base_url of '{provider_key}' is not an http(s) URL"}

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
        logger.warning("Voice registry request could not resolve the provider API key", exc_info=True)
        api_key = str(cfg.get("api_key") or "")
    return base_url, str(api_key or ""), None


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def _post_form(url: str, form: aiohttp.FormData, headers: Dict[str, str], timeout: float) -> Tuple[int, str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, data=form, headers=headers) as resp:
            return resp.status, await resp.text()


async def _http_get(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url, headers=headers) as resp:
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


async def register_voice_bytes(
    provider_key: str,
    block: Mapping[str, Any],
    *,
    sample: bytes,
    filename: str,
    name: Optional[str] = None,
    ref_text: str,
    consent: Optional[str] = None,
    env: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Upload *sample* (the bytes of *filename*) as voice *name* on the provider's server.

    Returns ``{"success": bool, "message": str}`` and, on success, ``voice``
    with the registered name, which the provider block can then carry as
    ``voice``.
    """
    base_url, api_key, failure = _resolve_endpoint(provider_key, block, env)
    if failure is not None:
        return failure

    try:
        original_name = _bare_name(filename, "File name")
        content_type = _content_type_for(original_name)
        voice_name = validate_voice_name(name or Path(original_name).stem)
    except VoiceRegistrationError as exc:
        return {"success": False, "message": str(exc)}
    if not sample:
        return {"success": False, "message": "The sample is empty"}
    if len(sample) > MAX_SAMPLE_BYTES:
        return {
            "success": False,
            "message": f"The sample is {len(sample) / 1048576:.1f} MB; the server accepts at most 10 MB",
        }
    transcript = str(ref_text or "").strip()
    if not transcript:
        return {
            "success": False,
            "message": "Transcript (ref_text) is required: without it the server stores the sample but does not clone from it",
        }
    consent_id = str(consent or "").strip() or DEFAULT_CONSENT

    form = aiohttp.FormData()
    form.add_field("audio_sample", sample, filename=original_name, content_type=content_type)
    form.add_field("name", voice_name)
    form.add_field("ref_text", transcript)
    form.add_field("consent", consent_id)

    url = f"{base_url}/audio/voices"
    host = url_host(url) or base_url
    try:
        status, text = await _post_form(url, form, _auth_headers(api_key), UPLOAD_TIMEOUT_SEC)
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
    registered = str(voice_info.get("name") or voice_name) if isinstance(voice_info, dict) else voice_name
    return {
        "success": True,
        "voice": registered,
        "message": (
            f"Voice '{registered}' registered at {host} from {original_name} "
            f"({len(sample) / 1024:.0f} KB, transcript {len(transcript)} chars). "
            f"Set voice: {registered} on the provider to use it."
        ),
    }


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
    """Register the sample *file* of the provider's voices directory (the scripted form)."""
    environment: Dict[str, str] = dict(os.environ)
    if env:
        environment.update({str(k): str(v) for k, v in env.items() if v is not None and str(v) != ""})
    cfg = substitute_env_vars(dict(block or {}), environment)
    try:
        path = resolve_sample_path(str(cfg.get("tts_voices_dir") or DEFAULT_VOICES_DIR), file)
    except VoiceRegistrationError as exc:
        return {"success": False, "message": str(exc)}
    return await register_voice_bytes(
        provider_key,
        block,
        sample=path.read_bytes(),
        filename=path.name,
        name=name,
        ref_text=ref_text,
        consent=consent,
        env=env,
    )


async def list_voices(
    provider_key: str,
    block: Mapping[str, Any],
    *,
    env: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """The voices the provider's server knows: registered ones in full, built-in ones by name."""
    base_url, api_key, failure = _resolve_endpoint(provider_key, block, env)
    if failure is not None:
        return failure
    url = f"{base_url}/audio/voices"
    host = url_host(url) or base_url
    try:
        status, text = await _http_get(url, _auth_headers(api_key), LIST_TIMEOUT_SEC)
    except aiohttp.ClientConnectorError as exc:
        return {"success": False, "message": f"Cannot connect to the speech server at {base_url}: {exc}"}
    except asyncio.TimeoutError:
        return {"success": False, "message": f"The speech server at {base_url} did not answer within {LIST_TIMEOUT_SEC:.0f} s"}
    except aiohttp.ClientError as exc:
        return {"success": False, "message": f"Listing voices at {base_url} failed: {exc}"}
    if status != 200:
        return {"success": False, "message": f"The speech server refused to list voices ({_server_error_text(status, text)})"}
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return {"success": False, "message": f"The speech server at {host} answered with something other than a voice list"}

    registered = []
    for entry in payload.get("uploaded_voices") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        registered.append(
            {
                "name": str(entry.get("name")),
                "created_at": entry.get("created_at"),
                "file_size": entry.get("file_size"),
                "ref_text": entry.get("ref_text"),
                "consent": entry.get("consent"),
                "speaker_description": entry.get("speaker_description"),
            }
        )
    builtin = [str(v) for v in (payload.get("voices") or []) if isinstance(v, (str, int))]
    return {
        "success": True,
        "voices": registered,
        "builtin": builtin,
        "message": f"{len(registered)} registered voice(s) on {host}"
        + (f"; built-in: {', '.join(builtin[:10])}" if builtin else ""),
    }
