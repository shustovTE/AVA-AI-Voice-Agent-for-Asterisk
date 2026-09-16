"""The LLM warm-up: one ``max_tokens: 1`` request with the first turn's prefix.

Each call has its own adapter and connection pool, so its first Chat
Completions request pays a new TCP+TLS connection and the prefill of the
whole system prompt. The warm-up sends exactly the prefix the first turn
will carry, while the greeting plays, through the adapter's own session; the
first turn then reuses the connection and, on an endpoint with prefix
caching, the cached prompt.
"""

import json

import aiohttp
import pytest

from src.config import OpenAIProviderConfig, split_openai_passthrough_fields
from src.pipelines import openai as openai_module
from src.pipelines.openai import OpenAILLMAdapter
from src.utils.http_trace import HttpTrace
from tests.test_pipeline_openai_transport import (
    _Lines,
    _Response,
    _Session,
    _adapter,
    _app_config,
    _stream_lines,
)

GREETING = "Здравствуйте! Это Анна из компании Домео."
CONTEXT = {"prior_messages": [{"role": "assistant", "content": GREETING}]}
OPTIONS = {
    "system_prompt": "Ты Анна, менеджер компании Домео.",
    "max_tokens": 200,
    "extra_body": {"prompt_cache_key": "prompt-1"},
}


def _completion(usage=None):
    body = {"choices": [{"message": {"content": "Да"}, "finish_reason": "length"}]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode("utf-8")


def _warm_up_then_stream(on_enter=None):
    """A session that answers the warm-up with a completion and the turn with a stream."""

    def _respond(kwargs):
        if kwargs["json"].get("stream"):
            return _Response(content=_Lines(_stream_lines()), on_enter=on_enter)
        return _Response(body=_completion(), on_enter=on_enter)

    return _Session(_respond)


class _Tool:
    class definition:
        @staticmethod
        def to_openai_schema():
            return {"type": "function", "function": {"name": "hangup", "parameters": {}}}


class _Registry:
    def get(self, name):
        return _Tool() if name == "hangup" else None


# --- what the warm-up sends ------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_up_sends_the_first_turns_prefix_with_one_token():
    session = _warm_up_then_stream()
    adapter = _adapter(provider={"warm_up": True}, session_factory=lambda: session)
    await adapter.start()

    summary = await adapter.warm_up("call-1", CONTEXT, OPTIONS)
    chunks = [c async for c in adapter.generate_stream("call-1", "Да, удобно.", CONTEXT, OPTIONS)]

    assert summary["status"] == "ok"
    assert "".join(chunks) == "Да, конечно."
    (warm_url, warm_kwargs), (turn_url, turn_kwargs) = session.posts
    warm, turn = warm_kwargs["json"], turn_kwargs["json"]
    assert warm_url == turn_url
    # Everything before the caller's words is identical: system prompt, greeting.
    assert warm["messages"][:-1] == turn["messages"][:-1]
    assert [m["role"] for m in warm["messages"]] == ["system", "assistant", "user"]
    assert warm["messages"][-1] == {"role": "user", "content": OpenAILLMAdapter.WARM_UP_USER_MESSAGE}
    assert turn["messages"][-1] == {"role": "user", "content": "Да, удобно."}
    # One token, no stream, and the same vendor fields (the cache key among them).
    assert warm["max_tokens"] == 1
    assert turn["max_tokens"] == 200
    assert "stream" not in warm and turn["stream"] is True
    assert warm["model"] == turn["model"]
    assert warm["prompt_cache_key"] == turn["prompt_cache_key"] == "prompt-1"
    assert warm_kwargs["headers"] == turn_kwargs["headers"]


@pytest.mark.asyncio
async def test_warm_up_advertises_the_same_tools_as_the_turn():
    session = _warm_up_then_stream()
    adapter = _adapter(provider={"warm_up": True}, session_factory=lambda: session)
    adapter.bind_tool_registry(_Registry())
    await adapter.start()
    options = {**OPTIONS, "tools": ["hangup", "unknown_tool"]}

    await adapter.warm_up("call-1", CONTEXT, options)
    [c async for c in adapter.generate_stream("call-1", "Алло", CONTEXT, options)]

    warm, turn = (kwargs["json"] for _url, kwargs in session.posts)
    assert warm["tools"] == turn["tools"] == [_Tool.definition.to_openai_schema()]
    assert warm["tool_choice"] == turn["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_warm_up_goes_through_the_configured_proxy():
    session = _warm_up_then_stream()
    adapter = _adapter(
        pipeline_options={"proxy": "http://u:p@10.0.0.5:8080", "warm_up": True},
        session_factory=lambda: session,
    )
    await adapter.start()

    await adapter.warm_up("call-1", CONTEXT, OPTIONS)

    _url, kwargs = session.posts[0]
    assert kwargs["proxy"] == "http://10.0.0.5:8080"
    assert kwargs["proxy_headers"]["Proxy-Authorization"].startswith("Basic ")
    assert kwargs["timeout"] == adapter._default_timeout


# --- what it reports -------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_up_logs_the_connection_and_the_cache_state(monkeypatch):
    infos = []
    monkeypatch.setattr(openai_module.logger, "info", lambda event, **kw: infos.append((event, kw)))
    usage = {"prompt_tokens": 812, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 768}}

    def _respond(kwargs):
        trace = kwargs["trace_request_ctx"]
        assert isinstance(trace, HttpTrace)

        def _hooks_ran():
            trace.connection = "new"
            trace.connect_ms = 143.2
            trace.headers_ms = 210.7

        return _Response(body=_completion(usage), on_enter=_hooks_ran)

    session = _Session(_respond)
    adapter = _adapter(provider={"warm_up": True}, session_factory=lambda: session)
    await adapter.start()
    await adapter._ensure_session()
    adapter._trace_enabled = True  # as _ensure_session sets it for a session of its own

    summary = await adapter.warm_up("call-1", CONTEXT, OPTIONS)

    completed = [kw for event, kw in infos if event == "LLM prompt warm-up completed"]
    assert len(completed) == 1
    assert completed[0]["call_id"] == "call-1"
    assert completed[0]["messages_count"] == 3
    assert completed[0]["tools_count"] == 0
    assert completed[0]["prompt_tokens"] == 812
    assert completed[0]["cached_tokens"] == 768
    assert completed[0]["connection"] == "new"
    assert completed[0]["connect_ms"] == 143.2
    assert completed[0]["total_ms"] >= 0
    assert summary["status"] == "ok"
    assert summary["cached_tokens"] == 768
    assert summary["connection"] == "new"


@pytest.mark.asyncio
async def test_warm_up_reports_missing_usage_as_unknown(monkeypatch):
    session = _Session(lambda kw: _Response(body=_completion()))
    adapter = _adapter(provider={"warm_up": True}, session_factory=lambda: session)
    await adapter.start()

    summary = await adapter.warm_up("call-1", CONTEXT, OPTIONS)

    assert summary["status"] == "ok"
    assert summary["prompt_tokens"] is None
    assert summary["cached_tokens"] is None


# --- it never hurts the call -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_warm_up_is_logged_and_swallowed(monkeypatch):
    warnings = []
    monkeypatch.setattr(openai_module.logger, "warning", lambda event, **kw: warnings.append((event, kw)))
    session = _Session(lambda kw: _Response(status=422, body=b'{"error": "max_tokens must be >= 8"}'))
    adapter = _adapter(provider={"warm_up": True}, session_factory=lambda: session)
    await adapter.start()

    summary = await adapter.warm_up("call-1", CONTEXT, OPTIONS)

    assert summary == {"status": "error", "http_status": 422, "total_ms": summary["total_ms"]}
    failed = [kw for event, kw in warnings if event == "LLM prompt warm-up failed"]
    assert len(failed) == 1
    assert failed[0]["status"] == 422
    assert "max_tokens" in failed[0]["body_preview"]


@pytest.mark.asyncio
async def test_a_connection_failure_during_warm_up_is_logged_and_swallowed(monkeypatch):
    warnings = []
    monkeypatch.setattr(openai_module.logger, "warning", lambda event, **kw: warnings.append((event, kw)))

    class _Refusing(_Session):
        def post(self, url, **kwargs):
            raise aiohttp.ClientConnectionError("connection refused")

    adapter = _adapter(provider={"warm_up": True}, session_factory=lambda: _Refusing(None))
    await adapter.start()

    summary = await adapter.warm_up("call-1", CONTEXT, OPTIONS)

    assert summary["status"] == "error"
    assert "refused" in summary["error"]
    assert [event for event, _kw in warnings] == ["LLM prompt warm-up failed"]


@pytest.mark.asyncio
async def test_warm_up_is_skipped_without_an_api_key_or_on_realtime():
    posted = []

    class _Recording(_Session):
        def post(self, url, **kwargs):
            posted.append(url)
            return _Response(body=_completion())

    keyless = OpenAILLMAdapter("native_llm", _app_config(), OpenAIProviderConfig(warm_up=True), {}, session_factory=lambda: _Recording(None))
    assert await keyless.warm_up("call-1", CONTEXT, OPTIONS) == {"status": "skipped", "reason": "no_api_key"}

    realtime = _adapter(provider={"warm_up": True}, session_factory=lambda: _Recording(None))
    assert await realtime.warm_up("call-1", CONTEXT, {**OPTIONS, "use_realtime": True}) == {"status": "skipped", "reason": "realtime"}

    assert posted == []


# --- the switch ------------------------------------------------------------------


def test_warm_up_is_off_unless_asked_for():
    assert _adapter().warm_up_enabled is False
    assert _adapter(provider={"warm_up": True}).warm_up_enabled is True
    # A pipeline's options.llm overrides the provider block, as for the other transport settings.
    assert _adapter(provider={"warm_up": True}, pipeline_options={"warm_up": False}).warm_up_enabled is False
    assert _adapter(pipeline_options={"warm_up": "yes"}).warm_up_enabled is True
    assert _adapter(pipeline_options={"warm_up": "0"}).warm_up_enabled is False
    assert _adapter(pipeline_options={"warm_up": "maybe"}).warm_up_enabled is False


def test_warm_up_never_reaches_the_request_body():
    assert split_openai_passthrough_fields({"warm_up": True, "prompt_cache_key": "k"}) == {"prompt_cache_key": "k"}
    adapter = _adapter(provider={"warm_up": True})
    payload = adapter._build_chat_payload("привет", {}, adapter._compose_options({}))
    assert "warm_up" not in payload
