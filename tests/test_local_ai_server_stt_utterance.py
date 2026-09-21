"""The local AI server decodes a whole utterance the client's VAD cut (``stt_utterance``).

One message, one final ``stt_result`` carrying the ``utterance_id`` back: no
voice activity detection, no idle finalizer and no echo guard in between.
A backend that takes only a stream answers with an error the client falls
back on.
"""

from __future__ import annotations

import base64
import importlib
import sys
from pathlib import Path

import pytest

LOCAL_AI_DIR = str(Path(__file__).resolve().parents[1] / "local_ai_server")


def _load(name: str):
    if LOCAL_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_AI_DIR)
    return importlib.import_module(name)


class _UtteranceBackend:
    def __init__(self, text="три комнаты"):
        self.text = text
        self.calls = []

    def transcribe_utterance(self, pcm16):
        self.calls.append(pcm16)
        return {"type": "final", "text": self.text}


def _server(backend_name="onnx_asr", backend=None):
    server_mod = _load("server")
    session_mod = _load("session")
    instance = object.__new__(server_mod.LocalAIServer)
    instance.mock_models = False
    instance.stt_backend = backend_name
    instance.onnx_asr_backend = backend if backend_name == "onnx_asr" else None
    instance.sherpa_backend = backend if backend_name == "sherpa" else None
    instance.sherpa_model_type = "offline"
    instance.faster_whisper_backend = None
    instance.whisper_cpp_backend = None
    instance.sent = []

    async def _send_json(_websocket, payload):
        instance.sent.append(payload)
        return True

    instance._send_json = _send_json
    return instance, session_mod.SessionContext(call_id="seed", mode="stt")


def _message(pcm16: bytes, **extra):
    return {"type": "stt_utterance", "mode": "stt", "call_id": "call-utt", "rate": 16000, "data": base64.b64encode(pcm16).decode("ascii"), **extra}


@pytest.mark.asyncio
async def test_an_utterance_is_decoded_whole_and_answered_with_one_final():
    backend = _UtteranceBackend()
    server, session = _server(backend=backend)
    pcm16 = b"\x01\x02" * 8000  # 500 ms

    await server._handle_stt_utterance(None, session, _message(pcm16, utterance_id="call-utt:utt-1"))

    assert backend.calls == [pcm16]
    assert server.sent == [
        {
            "type": "stt_result",
            "text": "три комнаты",
            "call_id": "call-utt",
            "mode": "stt",
            "is_final": True,
            "is_partial": False,
            "stt_backend": "onnx_asr",
            "utterance_id": "call-utt:utt-1",
        }
    ]
    assert session.stt_segmenter == "client" and session.utterances_decoded == 1
    assert session.last_final_text == "три комнаты"


@pytest.mark.asyncio
async def test_nothing_recognized_is_still_one_final_so_the_client_can_match_it():
    server, session = _server(backend=_UtteranceBackend(text="?"))

    await server._handle_stt_utterance(None, session, _message(b"\x00\x00" * 4000, utterance_id="u2"))

    assert server.sent[-1]["text"] == "" and server.sent[-1]["utterance_id"] == "u2"
    assert "error" not in server.sent[-1]


@pytest.mark.asyncio
async def test_a_backend_that_takes_only_a_stream_answers_with_an_error():
    server, session = _server(backend_name="tone")

    await server._handle_stt_utterance(None, session, _message(b"\x01\x02" * 4000, utterance_id="u3"))

    assert server.sent[-1]["error"] == "utterances_unsupported"
    assert server.sent[-1]["stt_utterances"] is False
    assert server.sent[-1]["utterance_id"] == "u3"
    assert server._stt_supports_utterances() is False


@pytest.mark.asyncio
async def test_a_decode_that_fails_is_reported_not_swallowed():
    class _Broken:
        def transcribe_utterance(self, pcm16):
            return None

    server, session = _server(backend=_Broken())
    await server._handle_stt_utterance(None, session, _message(b"\x01\x02" * 4000))
    assert server.sent[-1]["text"] == "" and server.sent[-1]["error"] == "utterance_decode_failed"


def test_the_server_says_whether_its_recognizer_decodes_utterances():
    server, _ = _server(backend=_UtteranceBackend())
    assert server._stt_supports_utterances() is True
    server_sherpa, _ = _server(backend_name="sherpa", backend=_UtteranceBackend())
    assert server_sherpa._stt_supports_utterances() is True
    server_sherpa.sherpa_model_type = "online"
    assert server_sherpa._stt_supports_utterances() is False
    server_mock, _ = _server(backend_name="tone")
    server_mock.mock_models = True
    assert server_mock._stt_supports_utterances() is True  # mock mode decodes anything as ""


@pytest.mark.asyncio
async def test_an_empty_payload_is_ignored():
    server, session = _server(backend=_UtteranceBackend())
    await server._handle_stt_utterance(None, session, {"type": "stt_utterance", "mode": "stt", "data": ""})
    assert server.sent == []
