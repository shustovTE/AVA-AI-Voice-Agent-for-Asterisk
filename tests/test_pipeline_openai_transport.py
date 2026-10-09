"""Transport settings of the OpenAI-compatible LLM adapter.

One Chat Completions request is made per caller turn, and the pause between
turns usually outlasts aiohttp's 15 s idle window, so every reply paid a
fresh TCP+TLS handshake. ``keepalive_timeout_sec`` widens that window on its
own, ``proxy`` routes the requests through an HTTP tunnel, the SSE body is
drained after ``[DONE]`` so the connection can be pooled, and every
completion log line says whether the connection was reused.
"""

import json

import aiohttp
import pytest

from src.config import AppConfig, OpenAIProviderConfig, split_openai_passthrough_fields
from src.pipelines import openai as openai_module
from src.pipelines.openai import OpenAILLMAdapter
from src.utils.http_trace import HttpTrace


def _app_config() -> AppConfig:
    return AppConfig(
        default_provider="native_llm",
        providers={"native_llm": {"api_key": "test-key"}},
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "persona", "model": "gpt-4o"},
        audio_transport="audiosocket",
        downstream_mode="stream",
    )


def _adapter(provider=None, pipeline_options=None, session_factory=None):
    provider_config = OpenAIProviderConfig(api_key="test-key", **(provider or {}))
    return OpenAILLMAdapter(
        "native_llm",
        _app_config(),
        provider_config,
        pipeline_options or {},
        session_factory=session_factory,
    )


class _Lines:
    """An SSE body: iterable lines plus the read() used to drain it."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.read_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)

    async def read(self):
        self.read_calls += 1
        self._lines.clear()
        return b""


class _Response:
    def __init__(self, *, status=200, body=b"", content=None, on_enter=None):
        self.status = status
        self._body = body
        self.content = content
        self._on_enter = on_enter

    async def __aenter__(self):
        if self._on_enter:
            self._on_enter()
        return self

    async def __aexit__(self, *_exc):
        return False

    async def text(self):
        return self._body.decode("utf-8")


class _Session:
    """Records every post; answers with a scripted response."""

    def __init__(self, response_factory):
        self._response_factory = response_factory
        self.posts = []
        self.closed = False

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self._response_factory(kwargs)

    async def close(self):
        self.closed = True


def _sse(chunk):
    return ("data: " + json.dumps(chunk) + "\n").encode("utf-8")


def _stream_lines():
    return [
        _sse({"choices": [{"delta": {"content": "Да, "}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "конечно."}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n",
    ]


class _FakeConnector:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeConnector.instances.append(self)


class _FakeClientSession:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        _FakeClientSession.instances.append(self)

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_aiohttp(monkeypatch):
    _FakeConnector.instances = []
    _FakeClientSession.instances = []
    monkeypatch.setattr(aiohttp, "TCPConnector", _FakeConnector)
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeClientSession)
    yield


# --- session construction -------------------------------------------------------


@pytest.mark.asyncio
async def test_keepalive_applies_without_a_proxy(fake_aiohttp):
    """The user's case: a longer idle window on the direct route."""
    adapter = _adapter(provider={"keepalive_timeout_sec": 180, "proxy": ""})

    await adapter._ensure_session()

    assert adapter._proxy_url is None
    assert _FakeConnector.instances[0].kwargs == {"keepalive_timeout": 180.0}
    session = _FakeClientSession.instances[0]
    assert session.kwargs["connector"] is _FakeConnector.instances[0]
    assert len(session.kwargs["trace_configs"]) == 1
    assert adapter._trace_enabled is True


@pytest.mark.asyncio
async def test_default_keepalive_keeps_aiohttp_defaults(fake_aiohttp):
    adapter = _adapter()

    await adapter._ensure_session()

    assert _FakeConnector.instances == []
    assert _FakeClientSession.instances[0].kwargs["connector"] is None


@pytest.mark.asyncio
async def test_transport_ready_is_logged_once_per_session(fake_aiohttp, monkeypatch):
    infos = []
    monkeypatch.setattr(openai_module.logger, "info", lambda event, **kw: infos.append((event, kw)))
    adapter = _adapter(pipeline_options={"proxy": "http://u:p@10.0.0.5:8080", "keepalive_timeout_sec": 120})

    await adapter._ensure_session()
    await adapter._ensure_session()

    ready = [kw for event, kw in infos if event == "OpenAI-compatible LLM transport ready"]
    assert len(ready) == 1
    assert ready[0]["proxy"] == "http://10.0.0.5:8080"
    assert ready[0]["proxy_authenticated"] is True
    assert ready[0]["keepalive_timeout_sec"] == 120.0


def test_pipeline_options_override_the_provider_transport():
    adapter = _adapter(
        provider={"proxy": "http://provider:8080", "keepalive_timeout_sec": 30},
        pipeline_options={"proxy": "http://pipeline:9090", "keepalive_timeout_sec": 240},
    )
    assert adapter._proxy_url == "http://pipeline:9090"
    assert adapter._keepalive_timeout_sec == 240.0


def test_unusable_transport_values_are_handled_like_elevenlabs():
    with pytest.raises(ValueError):
        _adapter(provider={"proxy": "socks5://10.0.0.5:1080"})
    # The typed provider field rejects text at config load; a pipeline option
    # is free-form and is ignored with a warning instead.
    assert _adapter(pipeline_options={"keepalive_timeout_sec": "soon"})._keepalive_timeout_sec is None
    assert _adapter(provider={"keepalive_timeout_sec": 0})._keepalive_timeout_sec is None


def test_transport_fields_never_reach_the_request_body():
    passthrough = split_openai_passthrough_fields(
        {"proxy": "http://x:8080", "keepalive_timeout_sec": 120, "prompt_cache_key": "k"}
    )
    assert passthrough == {"prompt_cache_key": "k"}

    adapter = _adapter(provider={"proxy": "http://x:8080", "keepalive_timeout_sec": 120})
    payload = adapter._build_chat_payload("привет", {}, adapter._compose_options({}))
    assert "proxy" not in payload
    assert "keepalive_timeout_sec" not in payload


# --- requests -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_credentials_become_a_header_on_every_request():
    session = _Session(lambda kw: _Response(body=json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()))
    adapter = _adapter(pipeline_options={"proxy": "http://u:p@10.0.0.5:8080"}, session_factory=lambda: session)

    await adapter.start()
    await adapter.generate("call-1", "hello", {}, {})

    _url, kwargs = session.posts[0]
    assert kwargs["proxy"] == "http://10.0.0.5:8080"
    assert kwargs["proxy_headers"]["Proxy-Authorization"].startswith("Basic ")
    assert kwargs["timeout"] == adapter._default_timeout
    # A session from a factory is not traced, so no tracing argument is sent.
    assert "trace_request_ctx" not in kwargs


@pytest.mark.asyncio
async def test_an_empty_proxy_means_a_direct_request():
    session = _Session(lambda kw: _Response(content=_Lines(_stream_lines())))
    adapter = _adapter(provider={"proxy": "", "keepalive_timeout_sec": 90}, session_factory=lambda: session)

    await adapter.start()
    chunks = [c async for c in adapter.generate_stream("call-1", "hello", {}, {})]

    assert "".join(chunks) == "Да, конечно."
    _url, kwargs = session.posts[0]
    assert "proxy" not in kwargs and "proxy_headers" not in kwargs
    assert adapter._keepalive_timeout_sec == 90.0


@pytest.mark.asyncio
async def test_the_stream_is_drained_after_done_and_completion_is_logged(monkeypatch):
    infos = []
    monkeypatch.setattr(openai_module.logger, "info", lambda event, **kw: infos.append((event, kw)))
    body = _Lines(_stream_lines() + [b"data: {\"late\": true}\n"])
    session = _Session(lambda kw: _Response(content=body))
    adapter = _adapter(session_factory=lambda: session)

    await adapter.start()
    chunks = [c async for c in adapter.generate_stream("call-1", "hello", {}, {})]

    assert "".join(chunks) == "Да, конечно."
    assert body.read_calls == 1
    completed = [kw for event, kw in infos if event == "OpenAI streaming completed"]
    assert len(completed) == 1
    assert completed[0]["finish_reason"] == "stop"
    assert completed[0]["chars"] == len("Да, конечно.")
    assert completed[0]["first_token_ms"] is not None
    assert completed[0]["total_ms"] >= completed[0]["first_token_ms"]


@pytest.mark.asyncio
async def test_a_body_without_read_does_not_break_the_stream():
    class _BareLines(_Lines):
        read = None

    session = _Session(lambda kw: _Response(content=_BareLines(_stream_lines())))
    adapter = _adapter(session_factory=lambda: session)

    await adapter.start()
    chunks = [c async for c in adapter.generate_stream("call-1", "hello", {}, {})]
    assert "".join(chunks) == "Да, конечно."


@pytest.mark.asyncio
async def test_completion_logs_carry_the_connection_state_when_traced(monkeypatch):
    """With a session the adapter built, the hooks fill the trace the log reports."""
    infos = []
    monkeypatch.setattr(openai_module.logger, "info", lambda event, **kw: infos.append((event, kw)))

    def _respond(kwargs):
        trace = kwargs["trace_request_ctx"]
        assert isinstance(trace, HttpTrace)

        def _hooks_ran():
            trace.connection = "new"
            trace.connect_ms = 143.2
            trace.headers_ms = 210.7

        return _Response(content=_Lines(_stream_lines()), on_enter=_hooks_ran)

    session = _Session(_respond)
    adapter = _adapter(session_factory=lambda: session)
    await adapter.start()
    await adapter._ensure_session()
    adapter._trace_enabled = True  # as _ensure_session sets it for a session of its own

    [c async for c in adapter.generate_stream("call-1", "hello", {}, {})]

    completed = [kw for event, kw in infos if event == "OpenAI streaming completed"][0]
    assert completed["connection"] == "new"
    assert completed["connect_ms"] == 143.2
    assert completed["headers_ms"] == 210.7
    assert session.posts[0][1]["trace_request_ctx"] is not None
