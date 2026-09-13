"""Fetch and verify pinned model files for the engine's local detectors.

Silero VAD and Smart Turn are small ONNX graphs the engine loads from the
``models/`` volume. Neither is bundled: a deployment either lets the engine
fetch the pinned release on first start, verifying its SHA-256, or places the
file with the matching ``scripts/fetch_*.sh`` on a host without outbound
access. Both paths land the same bytes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from typing import Any, Callable, Optional, Type

import structlog

logger = structlog.get_logger(__name__)


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    path: str,
    *,
    url: str,
    sha256: str,
    timeout: float = 60.0,
    opener: Optional[Callable[..., Any]] = None,
    error: Type[Exception] = RuntimeError,
) -> str:
    """Fetch ``url`` into ``path`` atomically, verifying its SHA-256.

    The download goes to a temporary file next to the destination and is
    renamed into place only after the digest matches, so a truncated or
    tampered transfer never leaves a half-written model behind.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise error(f"cannot create {directory} for the model file: {exc}") from exc
    open_url = opener or urllib.request.urlopen
    fd, temp_path = tempfile.mkstemp(prefix=".model.", suffix=".part", dir=directory)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out:
            with open_url(url, timeout=timeout) as response:
                for block in iter(lambda: response.read(1 << 16), b""):
                    out.write(block)
                    digest.update(block)
        actual = digest.hexdigest()
        if actual != sha256:
            raise error(f"download from {url} has sha256 {actual}, expected {sha256}")
        # The container user must be able to read what a host-side run fetched.
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except error:
        raise
    except Exception as exc:
        raise error(f"download from {url} failed: {exc}") from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return path


def ensure_file(
    path: str,
    *,
    url: str,
    sha256: str,
    version: str,
    auto_download: bool = True,
    timeout: float = 60.0,
    opener: Optional[Callable[..., Any]] = None,
    fetch_hint: str = "",
    error: Type[Exception] = RuntimeError,
    label: str = "model",
) -> str:
    """Return ``path`` once a usable file is there, fetching it if allowed.

    A present file is kept even when it is not the pinned release, so a
    deployment can drop in a different build on purpose; the mismatch is
    logged because a truncated copy would otherwise be hard to tell apart.
    """
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        try:
            actual = sha256_of_file(path)
        except OSError as exc:
            raise error(f"cannot read the {label} at {path}: {exc}") from exc
        if actual != sha256:
            logger.warning(
                "Model file is not the pinned release",
                label=label,
                path=path,
                sha256=actual,
                pinned_version=version,
                pinned_sha256=sha256,
            )
        return path
    if not auto_download:
        hint = f"; {fetch_hint}" if fetch_hint else ""
        raise error(f"{label} not found at {path}{hint}")
    logger.info("Fetching a model file", label=label, path=path, url=url, version=version)
    return download_file(path, url=url, sha256=sha256, timeout=timeout, opener=opener, error=error)
