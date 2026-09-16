"""
ElevenLabs TTS Pipeline Adapter.

Implements the TTSComponent interface for ElevenLabs text-to-speech API.

API Reference: https://elevenlabs.io/docs/api-reference/text-to-speech
"""
from __future__ import annotations

from ..audio.resampler import (
    alaw_to_pcm16le,
    convert_pcm16le_to_target_format,
    mulaw_to_pcm16le,
    resample_audio,
    resolve_output_resampler_policy,
)
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

import aiohttp

from ..config import AppConfig, ElevenLabsProviderConfig
from ..logging_config import get_logger
from ..utils.http_trace import HttpTrace, build_trace_config
from ..utils.proxy_url import (  # noqa: F401 - re-exported
    SUPPORTED_PROXY_SCHEMES,
    sanitize_proxy_url,
    split_proxy_credentials,
)
from .base import TTSComponent

logger = get_logger(__name__)

# Output formats the adapter can decode itself, mapped to the sample rate
# ElevenLabs returns them at: 16-bit PCM at every rate the API offers, and the
# two 8 kHz telephony codecs. The API also offers mp3_* and opus_*, which need
# a decoder the engine does not ship; those are refused before a request is
# made rather than failing on the first audio chunk.
_OUTPUT_FORMAT_SAMPLE_RATES = {
    "pcm_8000": 8000,
    "pcm_16000": 16000,
    "pcm_22050": 22050,
    "pcm_24000": 24000,
    "pcm_32000": 32000,
    "pcm_44100": 44100,  # Pro tier or above on ElevenLabs
    "pcm_48000": 48000,
    "ulaw_8000": 8000,
    "alaw_8000": 8000,
}
_MULAW_OUTPUT_FORMATS = {"ulaw_8000"}
_ALAW_OUTPUT_FORMATS = {"alaw_8000"}
_COMPRESSED_OUTPUT_FORMAT_PREFIXES = ("mp3_", "opus_")


def _unsupported_output_format(output_format: str) -> RuntimeError:
    accepted = ", ".join(_OUTPUT_FORMAT_SAMPLE_RATES)
    reason = (
        "mp3 and opus need a decoder the engine does not ship"
        if str(output_format).startswith(_COMPRESSED_OUTPUT_FORMAT_PREFIXES)
        else "not an output format this adapter can decode"
    )
    return RuntimeError(
        f"Unsupported ElevenLabs TTS output format: {output_format} ({reason}); "
        f"use one of: {accepted}"
    )


def _decode_to_pcm16le(raw: bytes, output_format: str) -> bytes:
    """Bytes as ElevenLabs sent them to PCM16; per-sample, so safe chunk by chunk."""
    if output_format in _MULAW_OUTPUT_FORMATS:
        return mulaw_to_pcm16le(raw)
    if output_format in _ALAW_OUTPUT_FORMATS:
        return alaw_to_pcm16le(raw)
    return raw

# The proxy rules live in src.utils.proxy_url so the Admin UI's connection
# probe applies exactly what this adapter applies; they are re-exported here
# for the adapter's callers and tests.
_SUPPORTED_PROXY_SCHEMES = SUPPORTED_PROXY_SCHEMES


class ElevenLabsTTSAdapter(TTSComponent):
    """
    ElevenLabs TTS adapter for pipeline orchestrator.
    
    Converts ElevenLabs' native audio into the per-call transport format.
    """

    wideband_output_format = {
        "encoding": "linear16",
        "sample_rate": 16000,
        "options": {"output_format": "pcm_16000"},
    }

    def __init__(
        self,
        component_key: str,
        app_config: AppConfig,
        provider_config: ElevenLabsProviderConfig,
        options: Optional[Dict[str, Any]] = None,
        *,
        session_factory: Optional[Callable[[], aiohttp.ClientSession]] = None,
    ):
        self.component_key = component_key
        self._app_config = app_config
        self._provider_config = provider_config
        self._pipeline_defaults = options or {}
        self._session_factory = session_factory
        self._session: Optional[aiohttp.ClientSession] = None
        # Proxy and connection reuse describe the transport, not one utterance,
        # so they are resolved once here rather than per synthesize() call. A
        # malformed proxy raises now: silently going direct would leak traffic
        # the operator asked to be tunnelled.
        self._proxy_url, self._proxy_headers = split_proxy_credentials(
            self._setting("proxy")
        )
        self._keepalive_timeout_sec = self._resolve_keepalive_timeout()
        self._trace_enabled = False

    def _trace_kwargs(self, trace: HttpTrace) -> Dict[str, Any]:
        """Attach connection tracing to a request on a session this adapter built."""
        return {"trace_request_ctx": trace} if self._trace_enabled else {}

    def _trace_fields(self, trace: Optional[HttpTrace]) -> Dict[str, Any]:
        if trace is None or not self._trace_enabled:
            return {}
        return trace.as_log_fields()

    def _setting(self, key: str) -> Any:
        """Read one transport setting: pipeline options override the provider."""
        if key in self._pipeline_defaults:
            return self._pipeline_defaults[key]
        return getattr(self._provider_config, key, None)

    def _resolve_keepalive_timeout(self) -> Optional[float]:
        raw = self._setting("keepalive_timeout_sec")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring unparsable ElevenLabs keepalive_timeout_sec",
                component=self.component_key,
                value=repr(raw),
            )
            return None
        return value if value > 0 else None

    async def start(self) -> None:
        """Initialize the adapter."""
        logger.debug(
            "ElevenLabs TTS adapter initialized",
            component=self.component_key,
            voice_id=self._provider_config.voice_id,
            model_id=self._provider_config.model_id,
        )

    async def stop(self) -> None:
        """Cleanup adapter resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def open_call(self, call_id: str, options: Dict[str, Any]) -> None:
        """Prepare for a call (ensure HTTP session exists)."""
        await self._ensure_session()

    async def close_call(self, call_id: str) -> None:
        """Cleanup call resources (no per-call state)."""
        pass

    async def validate_connectivity(self, options: Dict[str, Any]) -> Dict[str, Any]:
        # Merge provider config into options so the base validator sees base_url.
        merged = self._compose_options(options or {})
        return await super().validate_connectivity(merged)

    async def synthesize(
        self,
        call_id: str,
        text: str,
        options: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech using ElevenLabs API.
        
        Args:
            call_id: Unique call identifier
            text: Text to synthesize
            options: Runtime options (can override defaults)
            
        Yields:
            Audio chunks in the negotiated per-call transport format.
        """
        if not text:
            return
            yield  # Makes this an async generator
            
        await self._ensure_session()
        merged = self._compose_options(options)
        
        api_key = merged.get("api_key")
        if not api_key:
            raise RuntimeError("ElevenLabs TTS requires an API key (ELEVENLABS_API_KEY)")
        
        voice_id = merged.get("voice_id", self._provider_config.voice_id)
        model_id = merged.get("model_id", self._provider_config.model_id)
        output_format = merged.get("output_format", "ulaw_8000")
        target_encoding = merged["format"]["encoding"]
        target_sample_rate = int(merged["format"]["sample_rate"])

        # The historical adapter requested native μ-law for 8 kHz calls.
        # A wideband call must request 16 kHz PCM so no 8 kHz bottleneck is
        # introduced before the AudioSocket boundary; any 8 kHz format
        # (μ-law, A-law or 8 kHz PCM) would be that bottleneck.
        if (
            target_encoding.lower() in {"linear16", "pcm16", "slin16"}
            and target_sample_rate >= 16000
            and _OUTPUT_FORMAT_SAMPLE_RATES.get(output_format) == 8000
        ):
            logger.debug(
                "ElevenLabs output format raised for a wideband call",
                call_id=call_id,
                requested=output_format,
                used="pcm_16000",
            )
            output_format = "pcm_16000"
        
        request_id = f"11labs-tts-{uuid.uuid4().hex[:12]}"

        source_rate = _OUTPUT_FORMAT_SAMPLE_RATES.get(output_format)
        if source_rate is None:
            raise _unsupported_output_format(output_format)

        # Stream only when the API already returns the call's transport rate.
        # resample_audio() keeps no state between invocations, so resampling
        # chunk by chunk would add an artifact at every chunk boundary; those
        # calls keep the buffered path below.
        use_stream = (
            bool(merged.get("stream", True)) and source_rate == target_sample_rate
        )

        # Build API URL
        # https://elevenlabs.io/docs/api-reference/text-to-speech
        base_url = merged.get("base_url", self._provider_config.base_url)
        url = f"{base_url}/text-to-speech/{voice_id}"
        if use_stream:
            url = f"{url}/stream"
        
        # Voice settings
        voice_settings = {
            "stability": merged.get("stability", self._provider_config.stability),
            "similarity_boost": merged.get("similarity_boost", self._provider_config.similarity_boost),
            "style": merged.get("style", self._provider_config.style),
            "use_speaker_boost": merged.get("use_speaker_boost", self._provider_config.use_speaker_boost),
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/*",
        }
        
        params = {
            "output_format": output_format,
        }
        
        logger.info(
            "ElevenLabs TTS synthesis started",
            call_id=call_id,
            request_id=request_id,
            text_preview=text[:64],
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            streamed=use_stream,
        )
        
        started_at = time.perf_counter()
        trace = HttpTrace()
        
        try:
            async with self._session.post(
                url,
                json=payload,
                headers=headers,
                params=params,
                proxy=self._proxy_url,
                proxy_headers=self._proxy_headers,
                **self._trace_kwargs(trace),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.error(
                        "ElevenLabs TTS synthesis failed",
                        call_id=call_id,
                        request_id=request_id,
                        status=response.status,
                        body=body,
                    )
                    response.raise_for_status()
                
                if use_stream:
                    async for frame in self._iter_streamed_frames(
                        response,
                        call_id=call_id,
                        request_id=request_id,
                        started_at=started_at,
                    trace=trace,
                        output_format=output_format,
                        target_encoding=target_encoding,
                        target_sample_rate=target_sample_rate,
                        chunk_ms=int(merged.get("chunk_size_ms", 20)),
                    ):
                        yield frame
                    return

                # Read the full audio response
                raw_audio = await response.read()
                latency_ms = (time.perf_counter() - started_at) * 1000.0

                pcm_audio = _decode_to_pcm16le(raw_audio, output_format)

                if source_rate != target_sample_rate:
                    pcm_audio, _ = resample_audio(
                        pcm_audio,
                        source_rate,
                        target_sample_rate,
                        mode=merged["output_resampler"],
                    )
                converted = convert_pcm16le_to_target_format(
                    pcm_audio, target_encoding
                )
                
                logger.info(
                    "ElevenLabs TTS synthesis completed",
                    call_id=call_id,
                    request_id=request_id,
                    latency_ms=round(latency_ms, 2),
                    raw_bytes=len(raw_audio),
                    output_bytes=len(converted),
                    target_encoding=target_encoding,
                    target_sample_rate=target_sample_rate,
                    streamed=False,
                    **self._trace_fields(trace),
                )
                
                # Yield in chunks for streaming playback
                chunk_ms = int(merged.get("chunk_size_ms", 20))
                for chunk in self._chunk_audio(
                    converted, target_encoding, target_sample_rate, chunk_ms
                ):
                    if chunk:
                        yield chunk
                        
        except aiohttp.ClientError as exc:
            logger.error(
                "ElevenLabs TTS HTTP error",
                call_id=call_id,
                request_id=request_id,
                error=str(exc),
            )
            raise

    async def _iter_streamed_frames(
        self,
        response: aiohttp.ClientResponse,
        *,
        call_id: str,
        request_id: str,
        started_at: float,
        output_format: str,
        target_encoding: str,
        target_sample_rate: int,
        chunk_ms: int,
        trace: Optional[HttpTrace] = None,
    ) -> AsyncIterator[bytes]:
        """Emit playback frames while the response body is still arriving.

        Both conversions used here (μ-law or A-law decode and target encoding)
        are per-sample and stateless, so converting each network chunk on its own
        produces the same bytes as converting the whole response at once. The
        caller only selects this path when no resampling is required.
        """
        frame_bytes = self._frame_size_bytes(
            target_encoding, target_sample_rate, chunk_ms
        )
        source_is_companded = (
            output_format in _MULAW_OUTPUT_FORMATS or output_format in _ALAW_OUTPUT_FORMATS
        )
        pending = b""
        partial_sample = b""
        raw_bytes = 0
        output_bytes = 0
        first_audio_ms: Optional[float] = None

        async for raw in response.content.iter_any():
            if not raw:
                continue
            raw_bytes += len(raw)
            if source_is_companded:
                pcm_audio = _decode_to_pcm16le(raw, output_format)
            else:
                # A PCM16 sample can straddle a chunk boundary; hold the odd
                # trailing byte back for the next chunk.
                buffered = partial_sample + raw
                aligned = len(buffered) - (len(buffered) % 2)
                partial_sample = buffered[aligned:]
                pcm_audio = buffered[:aligned]
            if not pcm_audio:
                continue

            converted = convert_pcm16le_to_target_format(pcm_audio, target_encoding)
            if not converted:
                continue

            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - started_at) * 1000.0
                logger.info(
                    "ElevenLabs TTS first audio chunk",
                    call_id=call_id,
                    request_id=request_id,
                    first_audio_ms=round(first_audio_ms, 2),
                )

            pending += converted
            while len(pending) >= frame_bytes:
                frame = pending[:frame_bytes]
                pending = pending[frame_bytes:]
                output_bytes += len(frame)
                yield frame

        if pending:
            output_bytes += len(pending)
            yield pending

        logger.info(
            "ElevenLabs TTS synthesis completed",
            call_id=call_id,
            request_id=request_id,
            latency_ms=round((time.perf_counter() - started_at) * 1000.0, 2),
            first_audio_ms=(
                round(first_audio_ms, 2) if first_audio_ms is not None else None
            ),
            raw_bytes=raw_bytes,
            output_bytes=output_bytes,
            target_encoding=target_encoding,
            target_sample_rate=target_sample_rate,
            streamed=True,
            **self._trace_fields(trace),
        )

    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if self._session and not self._session.closed:
            return
        if self._session_factory is not None:
            self._session = self._session_factory()
            self._trace_enabled = False
            return
        connector = (
            aiohttp.TCPConnector(keepalive_timeout=self._keepalive_timeout_sec)
            if self._keepalive_timeout_sec is not None
            else None
        )
        self._session = aiohttp.ClientSession(
            connector=connector, trace_configs=[build_trace_config()]
        )
        self._trace_enabled = True
        if self._proxy_url:
            logger.info(
                "ElevenLabs TTS routed through a proxy",
                component=self.component_key,
                proxy=self._proxy_url,
                proxy_authenticated=self._proxy_headers is not None,
                keepalive_timeout_sec=self._keepalive_timeout_sec,
            )

    def _compose_options(self, runtime_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge runtime options with defaults."""
        runtime_options = runtime_options or {}
        runtime_format = runtime_options.get("format") or runtime_options.get("target_format") or {}
        default_format = self._pipeline_defaults.get("format") or self._pipeline_defaults.get("target_format") or {}
        
        # Priority: runtime > pipeline defaults > provider config
        merged = {
            "api_key": runtime_options.get("api_key", 
                self._pipeline_defaults.get("api_key", self._provider_config.api_key)),
            "voice_id": runtime_options.get("voice_id",
                self._pipeline_defaults.get("voice_id", self._provider_config.voice_id)),
            "model_id": runtime_options.get("model_id",
                self._pipeline_defaults.get("model_id", self._provider_config.model_id)),
            "base_url": runtime_options.get("base_url",
                self._pipeline_defaults.get("base_url", self._provider_config.base_url)),
            "output_format": runtime_options.get("output_format",
                self._pipeline_defaults.get("output_format", self._provider_config.output_format)),
            "format": {
                "encoding": runtime_format.get(
                    "encoding", default_format.get("encoding", "mulaw")
                ),
                "sample_rate": int(
                    runtime_format.get(
                        "sample_rate", default_format.get("sample_rate", 8000)
                    )
                ),
            },
            "stability": runtime_options.get("stability",
                self._pipeline_defaults.get("stability", self._provider_config.stability)),
            "similarity_boost": runtime_options.get("similarity_boost",
                self._pipeline_defaults.get("similarity_boost", self._provider_config.similarity_boost)),
            "style": runtime_options.get("style",
                self._pipeline_defaults.get("style", self._provider_config.style)),
            "use_speaker_boost": runtime_options.get("use_speaker_boost",
                self._pipeline_defaults.get("use_speaker_boost", self._provider_config.use_speaker_boost)),
            "chunk_size_ms": runtime_options.get("chunk_size_ms",
                self._pipeline_defaults.get("chunk_size_ms", 20)),
            "stream": runtime_options.get("stream",
                self._pipeline_defaults.get(
                    "stream", getattr(self._provider_config, "stream", True)
                )),
            "output_resampler": runtime_options.get(
                "output_resampler",
                self._pipeline_defaults.get(
                    "output_resampler", self._provider_config.output_resampler
                ),
            ),
        }
        merged["output_resampler"] = resolve_output_resampler_policy(
            provider_mode=merged.get("output_resampler")
        )[0]
        return merged

    def _frame_size_bytes(
        self,
        encoding: str,
        sample_rate: int,
        chunk_ms: int,
    ) -> int:
        """Bytes carried by one playback frame in the transport encoding."""
        bytes_per_sample = 1 if encoding.lower() in {"ulaw", "mulaw", "mu-law"} else 2
        return max(
            bytes_per_sample,
            int(sample_rate * (chunk_ms / 1000.0) * bytes_per_sample),
        )

    def _chunk_audio(
        self,
        audio: bytes,
        encoding: str,
        sample_rate: int,
        chunk_ms: int = 20,
    ) -> list:
        """Split encoded audio into transport-sized playback chunks."""
        chunk_size = self._frame_size_bytes(encoding, sample_rate, chunk_ms)
        
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
