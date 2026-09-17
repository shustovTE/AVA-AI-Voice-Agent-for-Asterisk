from __future__ import annotations

from typing import Any, Dict, Optional

from backends.interface import STTBackendInterface


class OnnxAsrBackend(STTBackendInterface):
    """Registry entry for the onnx-asr offline backend (GigaAM v3, NeMo FastConformer RU).

    The runtime implementation is ``OnnxAsrSTTBackend`` in ``stt_backends.py``;
    this stub only advertises the backend and its configuration schema.
    """

    def __init__(self):
        self._model = None

    @classmethod
    def name(cls) -> str:
        return "onnx_asr"

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        return {
            "model": {
                "type": "string",
                "required": False,
                "description": "onnx-asr model name (gigaam-v3-e2e-ctc, gigaam-v3-rnnt, nemo-fastconformer-ru-ctc, ...)",
            },
            "model_path": {
                "type": "string",
                "required": False,
                "description": "Directory with the model files; empty downloads the model into the cache",
            },
            "quantization": {"type": "string", "required": False, "enum": ["", "int8"]},
            "device": {"type": "string", "required": False, "enum": ["auto", "cpu", "cuda"]},
        }

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnx_asr  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, config: Dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        self._model = None

    def process_audio(self, audio_bytes: bytes) -> Optional[str]:
        raise NotImplementedError("OnnxAsrBackend is a registry stub; use OnnxAsrSTTBackend from stt_backends.py")

    def status(self) -> Dict[str, Any]:
        return {"backend": "onnx_asr", "loaded": self._model is not None}
