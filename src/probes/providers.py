"""Provider connection probes, run where the calls run.

The Admin UI's *Test connection* used to run inside the Admin UI container.
In a split deployment that container sits in another Docker network than the
engine: a vLLM on the engine host's loopback, an OpenAI-compatible LLM on that
host, or an ElevenLabs leg tunnelled through a proxy only the host can reach
are alive for calls and dead from the UI, so the button reported them broken.
The probes live here, the engine runs them on request (``POST
/providers/test`` on its health server) and the Admin UI relays the result;
when the engine is down the UI runs the same code and says so.

Every probe returns ``{"success": bool, "message": str}``, plus ``proxy`` and
``latency_ms`` where they mean something. Credentials never appear in a
result. The module has no engine-only imports beyond the shared config and
proxy helpers, so the Admin UI backend can import it too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlparse

import aiohttp

from ..config.provider_instances import resolve_secret_value
from ..utils.proxy_url import sanitize_proxy_url, split_proxy_credentials

logger = logging.getLogger(__name__)

ProbeResult = Dict[str, Any]
EnvLookup = Callable[[str], str]

# SECURITY: Hardcoded base URLs for provider validation requests. Maps a
# hostname to its canonical base URL, so a raw user string is never forwarded
# to a vendor (SSRF through chat_base_url in the config).
SAFE_BASE_URLS: Dict[str, str] = {
    "api.telnyx.com": "https://api.telnyx.com/v2/ai",
    "api.openai.com": "https://api.openai.com/v1",
    "api.groq.com": "https://api.groq.com/openai/v1",
    "openrouter.ai": "https://openrouter.ai/api/v1",
    "api.deepseek.com": "https://api.deepseek.com/v1",
    "api.minimax.io": "https://api.minimax.io/v1",
    "api.minimaxi.com": "https://api.minimaxi.com/v1",
    "api.anthropic.com": "https://api.anthropic.com/v1",
    "api.deepgram.com": "https://api.deepgram.com/v1",
    "api.elevenlabs.io": "https://api.elevenlabs.io/v1",
    "generativelanguage.googleapis.com": "https://generativelanguage.googleapis.com/v1beta",
}

OPENAI_SPEECH_ROUTES = ("/audio/speech", "/audio/transcriptions")

_ENV_PATTERN = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)(?::?[-=]([^}]*))?\}")
_AZURE_REGION_RE = re.compile(r"^[a-z][a-z0-9-]{0,48}[a-z0-9]$")


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------


def url_host(url: str) -> str:
    try:
        return (urlparse(str(url)).hostname or "").lower()
    except Exception:
        return ""


def safe_base_url(user_url: str, fallback: str) -> str:
    """Return the hardcoded base URL of a known vendor host, or *fallback*.

    Only the hostname of *user_url* is used for the lookup; the returned
    string is never derived from the user's value.
    """
    return SAFE_BASE_URLS.get(url_host(user_url), fallback)


def sanitized_http_url(user_url: str) -> str:
    """Rebuild an operator-configured http(s) URL from its parsed components.

    Returns "" when the value cannot serve as a verification target. The
    result carries no credentials, query or fragment, so a probe can only
    reach the scheme, host, port and path the operator configured.
    """
    try:
        parsed = urlparse(str(user_url or "").strip())
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https"):
        return ""
    host = (parsed.hostname or "").lower()
    if not host or parsed.username or parsed.password:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = f"{host}:{int(port)}" if port else host
    path = (parsed.path or "").rstrip("/")
    return f"{parsed.scheme}://{netloc}{path}"


def openai_probe_base_url(block: Mapping[str, Any]) -> str:
    """The base URL an OpenAI-compatible block is verified at, as configured.

    A chat block names it directly. A speech-only block (``fish_tts``,
    ``whisper_stt``) names only its route, so the ``/audio/speech`` or
    ``/audio/transcriptions`` tail is dropped and that host's own ``/models``
    is asked, instead of api.openai.com, where such a block never sends a
    request. Returns "" when the block names no endpoint at all.
    """
    if not isinstance(block, Mapping):
        return ""
    for key in ("chat_base_url", "base_url"):
        value = str(block.get(key) or "").strip()
        if value:
            return value
    for key in ("tts_base_url", "stt_base_url"):
        value = str(block.get(key) or "").strip().rstrip("/")
        if not value:
            continue
        for route in OPENAI_SPEECH_ROUTES:
            if value.lower().endswith(route):
                value = value[: -len(route)]
                break
        return value
    return ""


def saved_provider_base_host(saved_providers: Mapping[str, Any], provider_key: str) -> str:
    """Host of the endpoint saved on disk for *provider_key* (chat, else speech)."""
    if not provider_key or not isinstance(saved_providers, Mapping):
        return ""
    block = saved_providers.get(provider_key)
    if not isinstance(block, Mapping):
        for key, value in saved_providers.items():
            if str(key).lower() == provider_key.lower() and isinstance(value, Mapping):
                block = value
                break
    if not isinstance(block, Mapping):
        return ""
    return url_host(openai_probe_base_url(block))


def verification_base_url(
    user_url: str,
    fallback: str,
    *,
    provider_key: str = "",
    saved_providers: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve the base URL a provider verification request may call.

    Known vendor hosts keep their hardcoded canonical URL. A self-hosted
    OpenAI-compatible endpoint (vLLM, LiteLLM, llama.cpp, ...) is absent from
    that table; probing the vendor instead would leak the self-hosted token
    to a third party and report a meaningless 401. Such a host is accepted
    only when it matches the provider block already saved on disk, so a
    request body alone can never steer the probe at an arbitrary address. An
    empty return means the caller must not send credentials anywhere.
    """
    host = url_host(user_url)
    if host in SAFE_BASE_URLS:
        return SAFE_BASE_URLS[host]
    if not str(user_url or "").strip():
        return fallback
    if not host or host != saved_provider_base_host(saved_providers or {}, provider_key):
        return ""
    return sanitized_http_url(user_url)


def llm_legacy_env_names(provider_key: str, kind: str) -> tuple:
    prefix = provider_key.rsplit("_llm", 1)[0].upper()
    names = [f"{prefix}_API_KEY"]
    canonical = {
        "google": "GOOGLE_API_KEY",
        "telnyx": "TELNYX_API_KEY",
        "telenyx": "TELNYX_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }.get(kind)
    if canonical and canonical not in names:
        names.append(canonical)
    return tuple(names)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def substitute_env_vars(item: Any, env: Mapping[str, str]) -> Any:
    """Expand ``${VAR}``, ``${VAR:-default}`` and ``${VAR:=default}`` in config values."""
    if isinstance(item, dict):
        return {key: substitute_env_vars(value, env) for key, value in item.items()}
    if isinstance(item, list):
        return [substitute_env_vars(value, env) for value in item]
    if isinstance(item, str):

        def _replace(match: "re.Match[str]") -> str:
            value = env.get(match.group(1))
            if value:
                return str(value)
            default = match.group(2)
            return default if default is not None else ""

        return _ENV_PATTERN.sub(_replace, item)
    return item


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass
class HttpResponse:
    status: int
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


async def _http_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    proxy: Optional[str] = None,
    proxy_headers: Optional[Dict[str, str]] = None,
) -> HttpResponse:
    """One request with a fresh session; ``trust_env`` stays off, as in the adapters."""
    kwargs: Dict[str, Any] = {"headers": headers or {}}
    if json_body is not None:
        kwargs["json"] = json_body
    if proxy:
        kwargs["proxy"] = proxy
        if proxy_headers:
            kwargs["proxy_headers"] = proxy_headers
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.request(method, url, **kwargs) as resp:
            return HttpResponse(resp.status, await resp.text())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def probe_provider(
    name: str,
    config: Mapping[str, Any],
    *,
    saved_providers: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, Optional[str]]] = None,
) -> ProbeResult:
    """Test one provider block as the calls would reach it.

    *saved_providers* are the blocks saved on disk, which decide whether a
    self-hosted host may be probed at all. *env* overlays the process
    environment (the Admin UI passes its ``.env``).
    """
    environment: Dict[str, str] = dict(os.environ)
    if env:
        environment.update({str(k): str(v) for k, v in env.items() if v is not None and str(v) != ""})

    def get_env(key: str) -> str:
        return str(environment.get(key) or "")

    provider_config = substitute_env_vars(dict(config or {}), environment)
    provider_name = str(name or "").lower()
    saved = saved_providers if isinstance(saved_providers, Mapping) else {}
    try:
        return await _probe(provider_name, name or "", provider_config, get_env, saved)
    except asyncio.TimeoutError:
        return {"success": False, "message": "Connection timeout"}
    except Exception as exc:  # the result must always be a verdict, never a stack trace
        logger.debug("Provider probe failed", exc_info=True)
        return {"success": False, "message": f"Test failed: {exc}"}


async def _probe(
    provider_name: str,
    provider_key: str,
    provider_config: Dict[str, Any],
    get_env: EnvLookup,
    saved_providers: Mapping[str, Any],
) -> ProbeResult:
    # A saved provider may keep its credential in an owner-only file. Resolve
    # it in memory for this verification; it is never written back.
    if provider_config.get("api_key_file") or provider_config.get("api_key_env"):
        try:
            kind = str(provider_config.get("type") or provider_name.rsplit("_llm", 1)[0]).lower()
            resolved = resolve_secret_value(
                provider_config,
                file_field="api_key_file",
                env_field="api_key_env",
                inline_field="api_key",
                legacy_env_names=llm_legacy_env_names(provider_name, kind),
            )
            if resolved:
                provider_config["api_key"] = resolved
        except Exception:
            logger.warning("Provider connection test could not resolve managed API key")

    if "local" in provider_name or provider_config.get("type") == "local":
        return await _probe_local(provider_config, get_env)

    # ElevenLabs (full agent, or a modular TTS provider of type elevenlabs
    # whatever its name) before the other providers: the probe takes the
    # configured proxy exactly as the adapter does.
    if (
        "elevenlabs" in provider_name
        or "agent_id" in provider_config
        or str(provider_config.get("type") or "").lower() == "elevenlabs"
    ):
        return await _probe_elevenlabs(provider_config, get_env)

    if "realtime_base_url" in provider_config or "turn_detection" in provider_config:
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            return {"success": False, "message": "OPENAI_API_KEY not set in .env file"}
        response = await _http_request(
            "GET", "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"}
        )
        if response.status == 200:
            return {"success": True, "message": f"Connected to OpenAI (HTTP {response.status})"}
        return {"success": False, "message": f"OpenAI API error: HTTP {response.status}"}

    provider_type = str(provider_config.get("type") or "").lower()
    chat_base_url = str(provider_config.get("chat_base_url") or provider_config.get("base_url") or "").rstrip("/")
    host = url_host(chat_base_url)
    is_telnyx = provider_type in ("telnyx", "telenyx") or ("telnyx" in provider_name) or host == "api.telnyx.com"
    if is_telnyx:
        return await _probe_telnyx(provider_config, chat_base_url, get_env)

    if provider_type == "openai":
        return await _probe_openai_compatible(provider_name, provider_key, provider_config, get_env, saved_providers)

    if provider_type == "groq":
        return await _probe_groq(provider_config, get_env)

    if "google_live" in provider_config or (
        "llm_model" in provider_config and "gemini" in str(provider_config.get("llm_model", ""))
    ):
        api_key = get_env("GOOGLE_API_KEY")
        if not api_key:
            return {"success": False, "message": "GOOGLE_API_KEY not set in .env file"}
        response = await _http_request(
            "GET", f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        )
        if response.status == 200:
            return {"success": True, "message": f"Connected to Google API (HTTP {response.status})"}
        return {"success": False, "message": f"Google API error: HTTP {response.status}"}

    if "ws_url" in provider_config:
        return await _probe_websocket(str(provider_config.get("ws_url") or ""))

    if "ollama" in provider_name or provider_type == "ollama":
        return await _probe_ollama(provider_config)

    if any(key in provider_config for key in ("model", "stt_model", "chat_model", "tts_model")):
        return await _probe_by_model(provider_name, provider_config, get_env)

    if provider_type == "azure" or "azure" in provider_name:
        return await _probe_azure(provider_config, get_env)

    return {"success": False, "message": "Unknown provider type - cannot test"}


# ---------------------------------------------------------------------------
# Local AI server
# ---------------------------------------------------------------------------


async def _probe_local(provider_config: Dict[str, Any], get_env: EnvLookup) -> ProbeResult:
    import websockets

    # A leftover ${...} placeholder means nothing resolved; prefer the
    # operator's own .env values over a hardcoded loopback guess.
    ws_url = str(provider_config.get("base_url") or provider_config.get("ws_url") or "").strip()
    if not ws_url or "${" in ws_url or not ws_url.startswith(("ws://", "wss://")):
        ws_url = get_env("HEALTH_CHECK_LOCAL_AI_URL") or get_env("LOCAL_WS_URL") or "ws://127.0.0.1:8765"

    # local-ai-server rejects every message before auth once
    # LOCAL_WS_AUTH_TOKEN is set, so the probe authenticates the same way the
    # engine does; without it a healthy server reported "status invalid".
    auth_token = str(provider_config.get("auth_token") or "").strip()
    if not auth_token:
        auth_token = (get_env("LOCAL_WS_AUTH_TOKEN") or "").strip()

    declared_caps = [
        str(cap).strip().lower() for cap in (provider_config.get("capabilities") or []) if str(cap).strip()
    ]

    def _fallback_ws_url(url: str) -> str:
        # With host networking the compose name local_ai_server does not
        # resolve; loopback is the best-compatibility fallback.
        if "local_ai_server" in url:
            return url.replace("local_ai_server", "127.0.0.1")
        return url

    async def _try_connect(url: str) -> Dict[str, Any]:
        async with websockets.connect(url, open_timeout=5.0) as ws:
            if auth_token:
                await ws.send(json.dumps({"type": "auth", "auth_token": auth_token}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                auth_data = json.loads(raw)
                if auth_data.get("type") != "auth_response" or auth_data.get("status") != "ok":
                    raise PermissionError(str(auth_data.get("message") or "auth rejected"))
            await ws.send(json.dumps({"type": "status"}))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            return json.loads(response)

    try:
        try:
            data = await _try_connect(ws_url)
            effective_url = ws_url
        except PermissionError:
            raise
        except Exception:
            alt = _fallback_ws_url(ws_url)
            if alt == ws_url:
                raise
            data = await _try_connect(alt)
            effective_url = alt

        if data.get("type") == "status_response" and data.get("status") == "ok":
            models = data.get("models", {}) or {}
            stt_loaded = bool((models.get("stt") or {}).get("loaded", False))
            llm_loaded = bool((models.get("llm") or {}).get("loaded", False))
            tts_loaded = bool((models.get("tts") or {}).get("loaded", False))

            stt_backend = data.get("stt_backend", "unknown")
            tts_backend = data.get("tts_backend", "unknown")
            llm_path = (models.get("llm") or {}).get("path") or ""
            llm_model = llm_path.split("/")[-1] if llm_path else "none"

            status_parts = [
                f"STT: {stt_backend} ✓" if stt_loaded else "STT: not loaded",
                f"LLM: {llm_model} ✓" if llm_loaded else "LLM: not loaded",
                f"TTS: {tts_backend} ✓" if tts_loaded else "TTS: not loaded",
            ]

            # A modular provider owns only the capabilities it declares, and
            # LOCAL_AI_MODE=minimal deliberately skips the LLM preload.
            loaded = {"stt": stt_loaded, "llm": llm_loaded, "tts": tts_loaded}
            required = [cap for cap in declared_caps if cap in loaded] or ["stt", "llm", "tts"]
            runtime_mode = str(((data.get("config") or {}).get("runtime_mode")) or "").strip().lower()
            if runtime_mode == "minimal":
                required = [role for role in required if role != "llm"]
            missing = [role for role in required if not loaded[role]]

            if missing:
                return {
                    "success": False,
                    "message": (
                        f"Local AI Server connected ({effective_url}) but "
                        f"{', '.join(role.upper() for role in missing)} not loaded. "
                        f"{' | '.join(status_parts)}"
                    ),
                }
            return {
                "success": True,
                "message": f"Local AI Server connected ({effective_url}). {' | '.join(status_parts)}",
            }
        return {"success": False, "message": "Local AI Server responded but status invalid"}
    except PermissionError as exc:
        return {
            "success": False,
            "message": (
                f"Local AI Server rejected the auth token ({exc}). Check "
                "LOCAL_WS_AUTH_TOKEN in .env and auth_token on this provider."
            ),
        }
    except Exception as exc:
        logger.debug("Local AI Server validation failed: %s", exc, exc_info=True)
        return {"success": False, "message": f"Cannot connect to Local AI Server at {ws_url} (see server logs)"}


async def _probe_websocket(ws_url: str) -> ProbeResult:
    if not ws_url:
        return {"success": False, "message": "No WebSocket URL provided"}
    try:
        import websockets

        async with websockets.connect(ws_url, open_timeout=5.0) as ws:
            await ws.close()
        return {"success": True, "message": "Local AI server is reachable via WebSocket"}
    except ImportError:
        return {"success": False, "message": "websockets library not installed"}
    except Exception as exc:
        return {"success": False, "message": f"Cannot reach local AI server at {ws_url}. Error: {exc}"}


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------


async def _probe_elevenlabs(provider_config: Dict[str, Any], get_env: EnvLookup) -> ProbeResult:
    """Reach ElevenLabs the way the TTS adapter will: through ``proxy`` when set.

    The proxy value is parsed by the adapter's own rules (http/https only,
    inline credentials moved into a header), so a setting the adapter would
    refuse is reported here as well, and the outcome names which hop failed:
    the proxy itself, the tunnel through it, or ElevenLabs.
    """
    api_key = str(provider_config.get("api_key") or "").strip()
    if not api_key or "${" in api_key:
        api_key = get_env("ELEVENLABS_API_KEY")
    if not api_key:
        return {"success": False, "message": "ELEVENLABS_API_KEY not set in .env file"}

    proxy_setting = str(provider_config.get("proxy") or "").strip()
    try:
        proxy_url, proxy_headers = split_proxy_credentials(proxy_setting)
    except ValueError as exc:
        return {
            "success": False,
            "proxy": sanitize_proxy_url(proxy_setting),
            "message": f"Proxy setting rejected, as the engine would reject it: {exc}",
        }
    shown_proxy = sanitize_proxy_url(proxy_setting) if proxy_url else None
    route = f"via proxy {shown_proxy}" if shown_proxy else "directly"

    started = time.monotonic()
    try:
        response = await _http_request(
            "GET",
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key, "Accept": "application/json"},
            timeout=10.0,
            proxy=proxy_url,
            proxy_headers=proxy_headers,
        )
    except aiohttp.ClientProxyConnectionError as exc:
        return {"success": False, "proxy": shown_proxy, "message": f"Cannot connect to proxy {shown_proxy}: {exc}"}
    except aiohttp.ClientHttpProxyError as exc:
        return {
            "success": False,
            "proxy": shown_proxy,
            "message": f"Proxy {shown_proxy} refused the tunnel to api.elevenlabs.io: {exc}",
        }
    except aiohttp.ClientConnectorError as exc:
        target = f"proxy {shown_proxy}" if shown_proxy else "api.elevenlabs.io"
        return {"success": False, "proxy": shown_proxy, "message": f"Cannot connect to {target}: {exc}"}
    except asyncio.TimeoutError:
        return {"success": False, "proxy": shown_proxy, "message": f"Timed out reaching ElevenLabs {route}"}
    except aiohttp.ClientError as exc:
        return {"success": False, "proxy": shown_proxy, "message": f"ElevenLabs request failed {route}: {exc}"}

    latency_ms = int((time.monotonic() - started) * 1000)
    if response.status == 200:
        try:
            voice_count = len((response.json() or {}).get("voices", []))
        except Exception:
            voice_count = 0
        return {
            "success": True,
            "proxy": shown_proxy,
            "latency_ms": latency_ms,
            "message": f"Connected to ElevenLabs {route} ({voice_count} voices available, {latency_ms} ms)",
        }
    if response.status in (401, 403):
        return {
            "success": False,
            "proxy": shown_proxy,
            "latency_ms": latency_ms,
            "message": f"Reached ElevenLabs {route}, but the API key was rejected (HTTP {response.status})",
        }
    return {
        "success": False,
        "proxy": shown_proxy,
        "latency_ms": latency_ms,
        "message": f"ElevenLabs API error {route}: HTTP {response.status}",
    }


# ---------------------------------------------------------------------------
# Telnyx
# ---------------------------------------------------------------------------


def _telnyx_error_summary(response: HttpResponse) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("errors"), list) and payload["errors"]:
            first = payload["errors"][0] if isinstance(payload["errors"][0], dict) else {}
            parts = [p for p in (first.get("code"), first.get("title"), first.get("detail")) if p]
            if parts:
                return " / ".join(str(p) for p in parts)
    except Exception:
        pass
    text = (response.text or "").strip().replace("\n", " ")
    return text[:180] if text else f"HTTP {response.status}"


async def _probe_telnyx(provider_config: Dict[str, Any], chat_base_url: str, get_env: EnvLookup) -> ProbeResult:
    base_url = safe_base_url(chat_base_url, "https://api.telnyx.com/v2/ai")
    api_key = str(provider_config.get("api_key") or "").strip() or get_env("TELNYX_API_KEY")
    if not api_key:
        return {"success": False, "message": "TELNYX_API_KEY not set in .env"}

    model = str(provider_config.get("chat_model") or provider_config.get("model") or "").strip()
    if not model:
        model = "Qwen/Qwen3-235B-A22B"

    api_key_ref = str(provider_config.get("api_key_ref") or "").strip()
    if model.startswith("openai/") and not api_key_ref:
        return {
            "success": False,
            "message": "Telnyx external models like openai/* require api_key_ref (Integration Secret identifier).",
        }

    try:
        models_resp = await _http_request(
            "GET", f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20.0
        )
        if models_resp.status != 200:
            return {"success": False, "message": f"Telnyx /models failed: {_telnyx_error_summary(models_resp)}"}

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            "temperature": 0.0,
            "max_tokens": 16,
        }
        if api_key_ref:
            payload["api_key_ref"] = api_key_ref
        chat_resp = await _http_request(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json_body=payload,
            timeout=20.0,
        )
        if chat_resp.status == 200:
            return {"success": True, "message": f"Connected to Telnyx. Chat completion OK with model: {model}"}
        return {"success": False, "message": f"Telnyx chat completion failed: {_telnyx_error_summary(chat_resp)}"}
    except Exception as exc:
        logger.debug("Telnyx provider validation failed: %s", exc, exc_info=True)
        return {"success": False, "message": f"Cannot connect to Telnyx at {base_url} (see server logs)"}


# ---------------------------------------------------------------------------
# OpenAI-compatible, Groq, Ollama, Azure, by model name
# ---------------------------------------------------------------------------


async def _probe_openai_compatible(
    provider_name: str,
    provider_key: str,
    provider_config: Dict[str, Any],
    get_env: EnvLookup,
    saved_providers: Mapping[str, Any],
) -> ProbeResult:
    # A speech-only block (fish_tts, whisper_stt) has no chat_base_url; its
    # own host is probed at /models, never api.openai.com.
    chat_base_url = verification_base_url(
        openai_probe_base_url(provider_config),
        "https://api.openai.com/v1",
        provider_key=provider_key,
        saved_providers=saved_providers,
    )
    if not chat_base_url:
        return {
            "success": False,
            "message": (
                "Cannot verify this endpoint: chat_base_url (or, for a speech-only block, "
                "stt_base_url / tts_base_url) must be an http(s) URL, and a self-hosted host "
                "is only probed once the provider is saved with that same host."
            ),
        }
    api_key = str(provider_config.get("api_key") or "")
    if not api_key:
        inferred_env = None
        host = url_host(chat_base_url)
        if "groq" in provider_name or host == "api.groq.com":
            inferred_env = "GROQ_API_KEY"
        elif "openai" in provider_name or host == "api.openai.com":
            inferred_env = "OPENAI_API_KEY"
        if inferred_env:
            api_key = get_env(inferred_env)
    if not api_key:
        return {"success": False, "message": "API key missing for OpenAI-compatible provider (set api_key or env var)"}

    try:
        response = await _http_request(
            "GET", f"{chat_base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0
        )
    except Exception as exc:
        logger.debug("OpenAI-compatible provider validation failed: %s", exc, exc_info=True)
        return {"success": False, "message": f"Cannot connect to provider at {chat_base_url} (see server logs)"}
    if response.status == 200:
        try:
            models = response.json().get("data") or []
            return {"success": True, "message": f"Connected (OpenAI-compatible). Found {len(models)} models."}
        except Exception:
            return {"success": True, "message": f"Connected (OpenAI-compatible) (HTTP {response.status})"}
    if response.status == 401:
        return {"success": False, "message": "Invalid API key (401)"}
    return {"success": False, "message": f"Provider API error: HTTP {response.status}"}


async def _probe_groq(provider_config: Dict[str, Any], get_env: EnvLookup) -> ProbeResult:
    api_key = str(provider_config.get("api_key") or "") or get_env("GROQ_API_KEY")
    if not api_key:
        return {"success": False, "message": "GROQ_API_KEY not set (set api_key or env var)"}
    # SECURITY: pinned to the official Groq endpoint, never a user-provided base URL.
    base_url = "https://api.groq.com/openai/v1"
    try:
        response = await _http_request(
            "GET", f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0
        )
    except Exception as exc:
        logger.debug("Groq Speech provider validation failed: %s", exc, exc_info=True)
        return {"success": False, "message": f"Cannot connect to provider at {base_url} (see server logs)"}
    if response.status == 200:
        try:
            models = response.json().get("data") or []
            return {"success": True, "message": f"Connected (Groq Speech). Found {len(models)} models."}
        except Exception:
            return {"success": True, "message": f"Connected (Groq Speech) (HTTP {response.status})"}
    if response.status == 401:
        return {"success": False, "message": "Invalid API key (401)"}
    return {"success": False, "message": f"Provider API error: HTTP {response.status}"}


async def _probe_ollama(provider_config: Dict[str, Any]) -> ProbeResult:
    base_url = str(provider_config.get("base_url") or "http://localhost:11434").rstrip("/")
    try:
        response = await _http_request("GET", f"{base_url}/api/tags", timeout=10.0)
        if response.status == 200:
            models = response.json().get("models", []) or []
            return {"success": True, "message": f"Connected to Ollama! Found {len(models)} models."}
        return {"success": False, "message": f"Ollama returned status {response.status}"}
    except aiohttp.ClientConnectorError:
        return {
            "success": False,
            "message": f"Cannot connect to Ollama at {base_url}. Ensure Ollama is running and accessible.",
        }
    except asyncio.TimeoutError:
        return {"success": False, "message": "Connection timeout - is Ollama running?"}
    except Exception as exc:
        return {"success": False, "message": f"Ollama connection failed: {exc}"}


async def _probe_by_model(provider_name: str, provider_config: Dict[str, Any], get_env: EnvLookup) -> ProbeResult:
    if str(provider_config.get("model") or "").startswith("nova") or "deepgram" in provider_name:
        api_key = get_env("DEEPGRAM_API_KEY")
        if not api_key:
            return {"success": False, "message": "DEEPGRAM_API_KEY not set in .env file"}
        response = await _http_request(
            "GET", "https://api.deepgram.com/v1/projects", headers={"Authorization": f"Token {api_key}"}
        )
        if response.status == 200:
            return {"success": True, "message": f"Connected to Deepgram (HTTP {response.status})"}
        return {"success": False, "message": f"Deepgram API error: HTTP {response.status}"}

    api_key = get_env("OPENAI_API_KEY")
    if api_key:
        try:
            response = await _http_request(
                "GET", "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=5.0
            )
            if response.status == 200:
                return {"success": True, "message": f"Connected to OpenAI (HTTP {response.status})"}
        except Exception:
            pass
    return {"success": True, "message": "Provider configuration valid (No specific connection test available)"}


async def _probe_azure(provider_config: Dict[str, Any], get_env: EnvLookup) -> ProbeResult:
    api_key = get_env("AZURE_SPEECH_KEY")
    if not api_key:
        return {"success": False, "message": "AZURE_SPEECH_KEY not set in .env file"}
    region = str(provider_config.get("region") or "eastus").strip().lower()
    # The region becomes part of a hostname; reject anything but a plain label.
    if not region or not _AZURE_REGION_RE.match(region):
        return {
            "success": False,
            "message": f"Invalid Azure region '{region}'. Expected lowercase alphanumeric (e.g. 'eastus').",
        }
    token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    try:
        response = await _http_request(
            "POST", token_url, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=10.0
        )
    except Exception as exc:
        logger.debug("Azure Speech provider validation failed: %s", exc, exc_info=True)
        return {"success": False, "message": f"Cannot connect to Azure Speech Service at region '{region}' (see server logs)"}
    if response.status == 200:
        capabilities = provider_config.get("capabilities") or []
        cap_str = "/".join(str(c).upper() for c in capabilities) if capabilities else "Speech"
        return {"success": True, "message": f"Connected to Azure Speech Service ({region}). {cap_str} key valid."}
    if response.status == 401:
        return {"success": False, "message": "Invalid AZURE_SPEECH_KEY (401 Unauthorized)"}
    return {"success": False, "message": f"Azure Speech API returned HTTP {response.status} for region '{region}'"}
