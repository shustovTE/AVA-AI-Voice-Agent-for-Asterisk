"""Per-request connection tracing for the engine's aiohttp adapters.

Whether a request reused a pooled connection or paid for a new one (TCP,
TLS, and through a proxy the CONNECT round trip on top) is invisible in an
adapter's own timing: a slow first token and a cold connection look alike.
aiohttp reports the difference through :class:`aiohttp.TraceConfig` hooks;
this module turns them into a small per-request record the adapters attach
to their existing completion log lines, so ``connection=new connect_ms=143``
answers the question straight from the log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class HttpTrace:
    """What one request cost to get onto the wire; filled in by the hooks."""

    connection: str = "unknown"  # "new" (TCP+TLS just done) or "reused" (from the pool)
    connect_ms: Optional[float] = None  # TCP + TLS (+ proxy CONNECT) for a new connection
    dns_ms: Optional[float] = None  # host resolution, when it was not cached
    headers_ms: Optional[float] = None  # request start until response headers
    _started: float = 0.0
    _connect_started: float = 0.0
    _dns_started: float = 0.0

    def as_log_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {"connection": self.connection}
        if self.connect_ms is not None:
            fields["connect_ms"] = round(self.connect_ms, 1)
        if self.dns_ms is not None:
            fields["dns_ms"] = round(self.dns_ms, 1)
        if self.headers_ms is not None:
            fields["headers_ms"] = round(self.headers_ms, 1)
        return fields


def _trace_of(context: Any) -> Optional[HttpTrace]:
    trace = getattr(context, "trace_request_ctx", None)
    return trace if isinstance(trace, HttpTrace) else None


async def _on_request_start(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None:
        trace._started = time.perf_counter()


async def _on_connection_create_start(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None:
        trace.connection = "new"
        trace._connect_started = time.perf_counter()


async def _on_connection_create_end(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None and trace._connect_started:
        trace.connect_ms = (time.perf_counter() - trace._connect_started) * 1000.0


async def _on_connection_reuseconn(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None:
        trace.connection = "reused"


async def _on_dns_resolvehost_start(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None:
        trace._dns_started = time.perf_counter()


async def _on_dns_resolvehost_end(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None and trace._dns_started:
        trace.dns_ms = (time.perf_counter() - trace._dns_started) * 1000.0


async def _on_request_end(session, context, params) -> None:
    trace = _trace_of(context)
    if trace is not None and trace._started:
        trace.headers_ms = (time.perf_counter() - trace._started) * 1000.0


def build_trace_config() -> aiohttp.TraceConfig:
    """A TraceConfig whose hooks fill the :class:`HttpTrace` passed as ``trace_request_ctx``."""
    config = aiohttp.TraceConfig()
    config.on_request_start.append(_on_request_start)
    config.on_connection_create_start.append(_on_connection_create_start)
    config.on_connection_create_end.append(_on_connection_create_end)
    config.on_connection_reuseconn.append(_on_connection_reuseconn)
    config.on_dns_resolvehost_start.append(_on_dns_resolvehost_start)
    config.on_dns_resolvehost_end.append(_on_dns_resolvehost_end)
    config.on_request_end.append(_on_request_end)
    return config
