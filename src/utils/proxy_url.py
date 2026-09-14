"""Proxy URL handling shared by the engine's ElevenLabs adapter and the Admin UI probe.

Both must read one ``proxy`` setting the same way: only ``http://`` and
``https://`` are usable (aiohttp speaks no SOCKS), inline credentials become a
``Proxy-Authorization`` header rather than travelling in the URL, and a value
that cannot be used is an error, never a silent direct connection. Keeping the
rules here, with no engine imports, lets the Admin UI's "Test connection"
apply exactly what the adapter will apply.
"""

from __future__ import annotations

import base64
from typing import Dict, Optional, Tuple
from urllib.parse import unquote, urlsplit, urlunsplit

SUPPORTED_PROXY_SCHEMES = frozenset({"http", "https"})


def sanitize_proxy_url(proxy: str) -> str:
    """Return the proxy URL without credentials, safe to log and to show."""
    parts = urlsplit(proxy)
    if not parts.username and not parts.password:
        return proxy
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def split_proxy_credentials(
    proxy: Optional[str],
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Separate a proxy URL from any credentials written into it.

    aiohttp deprecated its ``proxy_auth`` argument, so inline credentials become
    a ``Proxy-Authorization`` header instead. Returns ``(None, None)`` when no
    proxy is configured and raises ``ValueError`` for a value the adapter could
    not use.
    """
    cleaned = (proxy or "").strip()
    if not cleaned:
        return None, None
    parts = urlsplit(cleaned)
    if parts.scheme not in SUPPORTED_PROXY_SCHEMES:
        raise ValueError(
            f"Unsupported ElevenLabs proxy scheme {parts.scheme!r}: aiohttp speaks "
            "only http:// and https://. Expose an HTTP inbound on the proxy, or "
            "install aiohttp-socks and route at the network level instead."
        )
    if not parts.hostname:
        raise ValueError(f"ElevenLabs proxy URL has no host: {cleaned!r}")
    if not parts.username and not parts.password:
        return cleaned, None
    token = base64.b64encode(
        f"{unquote(parts.username or '')}:{unquote(parts.password or '')}".encode("utf-8")
    ).decode("ascii")
    return sanitize_proxy_url(cleaned), {"Proxy-Authorization": f"Basic {token}"}
