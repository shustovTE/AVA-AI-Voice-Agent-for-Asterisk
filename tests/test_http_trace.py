"""Connection tracing for the aiohttp adapters.

The hooks fill an :class:`HttpTrace` handed to a request as
``trace_request_ctx``; anything else riding in that slot is left alone.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.utils import http_trace


def _context(trace):
    return SimpleNamespace(trace_request_ctx=trace)


@pytest.mark.asyncio
async def test_a_new_connection_is_reported_with_its_setup_time():
    trace = http_trace.HttpTrace()
    ctx = _context(trace)

    await http_trace._on_request_start(None, ctx, None)
    await http_trace._on_dns_resolvehost_start(None, ctx, None)
    await asyncio.sleep(0.01)
    await http_trace._on_dns_resolvehost_end(None, ctx, None)
    await http_trace._on_connection_create_start(None, ctx, None)
    await asyncio.sleep(0.01)
    await http_trace._on_connection_create_end(None, ctx, None)
    await http_trace._on_request_end(None, ctx, None)

    assert trace.connection == "new"
    assert trace.connect_ms is not None and trace.connect_ms >= 5
    assert trace.dns_ms is not None and trace.dns_ms >= 5
    assert trace.headers_ms is not None and trace.headers_ms >= trace.connect_ms
    fields = trace.as_log_fields()
    assert fields["connection"] == "new"
    assert set(fields) == {"connection", "connect_ms", "dns_ms", "headers_ms"}


@pytest.mark.asyncio
async def test_a_pooled_connection_is_reported_without_setup_time():
    trace = http_trace.HttpTrace()
    ctx = _context(trace)

    await http_trace._on_request_start(None, ctx, None)
    await http_trace._on_connection_reuseconn(None, ctx, None)
    await http_trace._on_request_end(None, ctx, None)

    assert trace.connection == "reused"
    assert trace.connect_ms is None
    assert "connect_ms" not in trace.as_log_fields()
    assert trace.as_log_fields()["connection"] == "reused"


@pytest.mark.asyncio
async def test_other_request_contexts_are_left_alone():
    ctx = _context({"not": "a trace"})
    await http_trace._on_connection_create_start(None, ctx, None)
    await http_trace._on_connection_reuseconn(None, ctx, None)
    await http_trace._on_request_end(None, ctx, None)
    assert ctx.trace_request_ctx == {"not": "a trace"}

    bare = _context(None)
    await http_trace._on_request_end(None, bare, None)


def test_trace_config_registers_every_hook():
    config = http_trace.build_trace_config()
    assert list(config.on_request_start) == [http_trace._on_request_start]
    assert list(config.on_connection_create_start) == [http_trace._on_connection_create_start]
    assert list(config.on_connection_create_end) == [http_trace._on_connection_create_end]
    assert list(config.on_connection_reuseconn) == [http_trace._on_connection_reuseconn]
    assert list(config.on_dns_resolvehost_start) == [http_trace._on_dns_resolvehost_start]
    assert list(config.on_dns_resolvehost_end) == [http_trace._on_dns_resolvehost_end]
    assert list(config.on_request_end) == [http_trace._on_request_end]


def test_unknown_trace_logs_only_the_connection_state():
    assert http_trace.HttpTrace().as_log_fields() == {"connection": "unknown"}
