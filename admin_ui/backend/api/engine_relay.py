"""Relay a request to the ai_engine health server, where it runs next to the calls.

The engine's health server accepts a request from localhost or with the
HEALTH_API_TOKEN. The Admin UI container reaches it over the compose network
when it can, and otherwise runs the request from inside the engine container
through the Docker socket, which is localhost for the engine. Both paths
return the engine's JSON; ``None`` means no path reached the engine at all.

A file upload takes the network path as multipart. When only the Docker
socket works, the file is first put into the engine container's staging
directory (``put_archive``), and the request then names it; a Docker exec
carries its body in an environment variable, which cannot hold a sample.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tarfile
import time
import uuid
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException

STAGING_PARENT = "/tmp"
STAGING_DIR_NAME = "ava-voice-uploads"

_EXEC_SCRIPT = r'''
import base64, json, os, sys, urllib.error, urllib.request

method = (os.environ.get("AVA_RELAY_METHOD") or "POST").upper()
body = base64.b64decode(os.environ.get("AVA_RELAY_BODY") or "") or None
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
        data=body if method != "GET" else None,
        headers={"Content-Type": "application/json"} if body and method != "GET" else {},
        method=method,
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


def _engine_container():
    import docker

    from api.system import _find_compose_service_container

    client = docker.from_env()
    return _find_compose_service_container(client, "ai_engine")


def _exec_request_sync(method: str, path: str, payload: Optional[Dict[str, Any]], timeout: float) -> Optional[Dict[str, Any]]:
    """Send the request from inside the engine container, so it is localhost there."""
    try:
        container = _engine_container()
        if not container:
            return None
        environment = {
            "AVA_RELAY_METHOD": method.upper(),
            "AVA_RELAY_BODY": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii") if payload is not None else "",
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


def _stage_sample_sync(filename: str, data: bytes) -> Optional[str]:
    """Put *data* into the engine container's staging directory; returns the staged name."""
    try:
        container = _engine_container()
        if not container:
            return None
        suffix = PurePosixPath(filename or "").suffix.lower() or ".wav"
        staged_name = f"{uuid.uuid4().hex}{suffix}"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            directory = tarfile.TarInfo(STAGING_DIR_NAME)
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o777
            directory.mtime = int(time.time())
            archive.addfile(directory)
            member = tarfile.TarInfo(f"{STAGING_DIR_NAME}/{staged_name}")
            member.size = len(data)
            member.mode = 0o644
            member.mtime = int(time.time())
            archive.addfile(member, io.BytesIO(data))
        if not container.put_archive(STAGING_PARENT, buffer.getvalue()):
            return None
        return staged_name
    except Exception:
        return None


def _decode_engine_reply(resp: httpx.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"AI Engine returned a non-JSON response: {resp.text[:200]}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="AI Engine returned an unexpected response")
    if resp.status_code >= 400 and "success" not in data:
        raise HTTPException(
            status_code=resp.status_code,
            detail=str(data.get("error") or data.get("message") or resp.text[:200]),
        )
    return data


async def _send_over_network(method: str, path: str, *, timeout: float, **request_kwargs) -> Optional[Dict[str, Any]]:
    """Try each engine address; ``None`` when none answers (or the token is refused)."""
    headers: Dict[str, str] = {}
    token = _health_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        for base in _engine_base_urls():
            try:
                resp = await client.request(method, f"{base}{path}", headers=headers, **request_kwargs)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                continue
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="AI Engine did not answer in time")
            except httpx.HTTPError:
                continue
            if resp.status_code == 403:
                return None
            return _decode_engine_reply(resp)
    return None


async def request_engine_json(
    method: str, path: str, payload: Optional[Dict[str, Any]] = None, *, timeout: float = 30.0
) -> Optional[Dict[str, Any]]:
    """Send a JSON request to the engine; ``None`` when the engine cannot be reached.

    An engine reply with a status of 400 or more that carries no ``success``
    verdict is raised as an HTTPException with the engine's text. A 403 (no
    token, or a wrong one) and an unreachable address fall through to the
    Docker-socket path, like the other engine relays of the Admin UI.
    """
    kwargs: Dict[str, Any] = {}
    if payload is not None and method.upper() != "GET":
        kwargs["json"] = payload
    result = await _send_over_network(method, path, timeout=timeout, **kwargs)
    if result is not None:
        return result
    return await asyncio.to_thread(_exec_request_sync, method, path, payload, timeout)


async def post_engine_json(path: str, payload: Dict[str, Any], *, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    return await request_engine_json("POST", path, payload, timeout=timeout)


async def get_engine_json(path: str, *, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    return await request_engine_json("GET", path, None, timeout=timeout)


async def post_engine_multipart(
    path: str,
    *,
    fields: Dict[str, str],
    sample: Tuple[str, bytes, str],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Upload *sample* (filename, bytes, content type) with *fields* to the engine.

    Over the network the file goes as multipart. Through the Docker socket
    it is staged into the engine container first, and the JSON request then
    names the staged file; the engine deletes it after reading.
    """
    filename, data, content_type = sample
    result = await _send_over_network(
        "POST",
        path,
        timeout=timeout,
        data=dict(fields),
        files={"audio_sample": (filename, data, content_type or "application/octet-stream")},
    )
    if result is not None:
        return result
    staged_name = await asyncio.to_thread(_stage_sample_sync, filename, data)
    if not staged_name:
        return None
    payload = {**fields, "staged_file": staged_name, "filename": filename}
    return await asyncio.to_thread(_exec_request_sync, "POST", path, payload, timeout)
