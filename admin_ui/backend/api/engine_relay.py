"""Relay a request to the ai_engine health server, where it runs next to the calls.

The engine's health server accepts a request from localhost or with the
HEALTH_API_TOKEN. The Admin UI container reaches it over the compose network
when it can, and otherwise runs the request from inside the engine container
through the Docker socket, which is localhost for the engine. Both paths
return the engine's JSON; ``None`` means no path reached the engine at all.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

_EXEC_SCRIPT = r'''
import base64, json, os, sys, urllib.error, urllib.request

body = base64.b64decode(os.environ["AVA_RELAY_BODY"])
path = os.environ["AVA_RELAY_PATH"]
timeout = float(os.environ.get("AVA_RELAY_TIMEOUT") or 30)
ports = []


def add_port(raw):
    try:
        port = int(str(raw).strip())
    except Exception:
        return
    if 1 <= port <= 65535 and port not in ports:
        ports.append(port)


add_port(os.getenv("HEALTH_BIND_PORT", ""))
try:
    import yaml
    for candidate in ("/app/config/ai-agent.local.yaml", "/app/config/ai-agent.yaml"):
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            add_port((cfg.get("health") or {}).get("port"))
        except Exception:
            pass
except Exception:
    pass
add_port(os.environ.get("AVA_RELAY_PORT", ""))
add_port(15000)

for port in ports:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            sys.stdout.write(resp.read().decode("utf-8", "replace"))
        sys.exit(0)
    except urllib.error.HTTPError as exc:
        sys.stdout.write(exc.read().decode("utf-8", "replace"))
        sys.exit(0)
    except Exception:
        continue
sys.exit(2)
'''


def _health_api_token() -> str:
    from api.system import _get_health_api_token

    return _get_health_api_token()


def _health_port() -> int:
    from api.system import _configured_ai_engine_health_port

    return _configured_ai_engine_health_port()


def _dotenv(key: str) -> str:
    from api.system import _dotenv_value

    return _dotenv_value(key) or ""


def _engine_base_urls() -> List[str]:
    """Candidate engine health-server bases, most specific first."""
    candidates: List[str] = []
    for raw in (
        os.getenv("HEALTH_CHECK_AI_ENGINE_URL") or _dotenv("HEALTH_CHECK_AI_ENGINE_URL"),
        os.getenv("AI_ENGINE_HEALTH_URL") or _dotenv("AI_ENGINE_HEALTH_URL"),
    ):
        raw = (raw or "").strip()
        if not raw:
            continue
        if raw.endswith("/health"):
            raw = raw[: -len("/health")]
        candidates.append(raw)
    port = _health_port()
    candidates.extend(
        [
            f"http://127.0.0.1:{port}",
            f"http://ai_engine:{port}",
            f"http://ai-engine:{port}",
            f"http://host.docker.internal:{port}",
        ]
    )
    out: List[str] = []
    for candidate in candidates:
        candidate = candidate.rstrip("/")
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _exec_post_sync(path: str, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """POST from inside the engine container, so the request is localhost there."""
    try:
        import docker

        from api.system import _find_compose_service_container

        client = docker.from_env()
        container = _find_compose_service_container(client, "ai_engine")
        if not container:
            return None
        environment = {
            "AVA_RELAY_BODY": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
            "AVA_RELAY_PATH": path,
            "AVA_RELAY_PORT": str(_health_port()),
            "AVA_RELAY_TIMEOUT": str(timeout),
        }
        code, out = container.exec_run(["python3", "-c", _EXEC_SCRIPT], environment=environment)
        if code != 0:
            return None
        raw = (out or b"").decode("utf-8", errors="replace").strip()
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def post_engine_json(path: str, payload: Dict[str, Any], *, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """POST *payload* to *path* on the engine; ``None`` when the engine cannot be reached.

    An engine reply with a status of 400 or more that carries no ``success``
    verdict is raised as an HTTPException with the engine's text, so the
    browser sees why. A 403 (no token, or a wrong one) falls through to the
    Docker-socket path, like the other engine relays of the Admin UI.
    """
    headers: Dict[str, str] = {}
    token = _health_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for base in _engine_base_urls():
            try:
                resp = await client.post(f"{base}{path}", headers=headers, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                continue
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="AI Engine did not answer in time")
            except httpx.HTTPError:
                continue
            if resp.status_code == 403:
                break
            try:
                data = resp.json()
            except ValueError:
                raise HTTPException(
                    status_code=502, detail=f"AI Engine returned a non-JSON response: {resp.text[:200]}"
                )
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="AI Engine returned an unexpected response")
            if resp.status_code >= 400 and "success" not in data:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=str(data.get("error") or data.get("message") or resp.text[:200]),
                )
            return data

    return await asyncio.to_thread(_exec_post_sync, path, payload, timeout)
