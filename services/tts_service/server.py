"""
TTS Service - gRPC Server

Text-to-Speech microservice that wraps TTS providers.
"""

import asyncio
import os
import sys
from concurrent import futures
from typing import AsyncIterator

import grpc
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.generated import tts_pb2, tts_pb2_grpc, common_pb2
from src.providers.base import TTSProvider, SynthesisResult as ProviderSynthesisResult
from src.providers.tts.piper_provider import PiperProvider


def synthesis_to_proto(result: ProviderSynthesisResult) -> common_pb2.SynthesisResult:
    """Convert provider SynthesisResult to proto"""
    return common_pb2.SynthesisResult(
        audio_data=result.audio_data,
        sample_rate=result.sample_rate,
        text=result.text,
        voice_id=result.voice_id,
        timestamp=result.timestamp,
        duration_ms=result.duration_ms,
        format=result.format,
    )


class TTSServicer(tts_pb2_grpc.TTSServiceServicer):
    """TTS Service implementation"""

    def __init__(self, provider: TTSProvider):
        self.provider = provider
        logger.info(f"TTS Service initialized with provider: {provider.__class__.__name__}")

    async def Synthesize(
        self,
        request: tts_pb2.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.SynthesisResult:
        """Synthesize speech from a single text"""
        try:
            logger.debug(f"Synthesize request: text={request.text[:50]}..., voice={request.voice_id}")

            # Build kwargs from request parameters
            kwargs = {}
            if request.HasField("speed"):
                kwargs["speed"] = request.speed
            if request.HasField("pitch"):
                kwargs["pitch"] = request.pitch
            kwargs.update(dict(request.parameters))

            # Synthesize
            result = await self.provider.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                language=request.language if request.language else None,
                **kwargs,
            )

            # Convert to proto
            return synthesis_to_proto(result)

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Synthesis failed: {str(e)}")

    async def SynthesizeStream(
        self,
        request_iterator: AsyncIterator[tts_pb2.SynthesizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.SynthesisResult]:
        """Synthesize speech from text stream"""
        try:
            logger.debug("SynthesizeStream started")

            # Extract first request for voice_id and other parameters
            first_request = await request_iterator.__anext__()
            voice_id = first_request.voice_id
            language = first_request.language if first_request.language else None

            # Build kwargs
            kwargs = {}
            if first_request.HasField("speed"):
                kwargs["speed"] = first_request.speed
            if first_request.HasField("pitch"):
                kwargs["pitch"] = first_request.pitch
            kwargs.update(dict(first_request.parameters))

            # Create text stream
            async def text_stream():
                yield first_request.text
                async for req in request_iterator:
                    yield req.text

            # Stream synthesis
            async for result in self.provider.synthesize_stream(
                text_stream(), voice_id, language, **kwargs
            ):
                yield synthesis_to_proto(result)

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Streaming synthesis failed: {str(e)}")

    async def GetVoices(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> tts_pb2.VoiceListResponse:
        """Get available voices"""
        try:
            voices_data = await self.provider.get_voices()

            voices = []
            for voice_dict in voices_data:
                voice = tts_pb2.Voice(
                    id=voice_dict.get("id", ""),
                    name=voice_dict.get("name", ""),
                    language=voice_dict.get("language", ""),
                    supported_languages=voice_dict.get("supported_languages", []),
                    gender=voice_dict.get("gender", "neutral"),
                    metadata=voice_dict.get("metadata", {}),
                )
                voices.append(voice)

            return tts_pb2.VoiceListResponse(voices=voices)

        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetSupportedLanguages(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> tts_pb2.LanguageListResponse:
        """Get supported languages"""
        try:
            languages = self.provider.supported_languages()
            return tts_pb2.LanguageListResponse(
                language_codes=languages,
                supports_streaming=self.provider.supports_streaming(),
                supports_voice_cloning=self.provider.supports_voice_cloning(),
            )
        except Exception as e:
            logger.error(f"Error getting supported languages: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def HealthCheck(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.HealthCheckResponse:
        """Health check"""
        try:
            healthy = await self.provider.health_check()
            status = self.provider.status.value

            return common_pb2.HealthCheckResponse(
                healthy=healthy,
                status=status,
                provider_name=self.provider.__class__.__name__,
                details={
                    "supports_streaming": str(self.provider.supports_streaming()),
                    "supports_voice_cloning": str(self.provider.supports_voice_cloning()),
                },
            )
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return common_pb2.HealthCheckResponse(
                healthy=False,
                status="error",
                provider_name=self.provider.__class__.__name__,
                details={"error": str(e)},
            )

    async def GetMetrics(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.MetricsResponse:
        """Get metrics"""
        try:
            metrics = self.provider.get_metrics()
            return common_pb2.MetricsResponse(
                total_requests=metrics.get("total_requests", 0),
                total_errors=metrics.get("total_errors", 0),
                avg_duration_ms=metrics.get("avg_duration_ms", 0.0),
                total_duration_ms=metrics.get("total_duration_ms", 0.0),
            )
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def load_provider() -> TTSProvider:
    """Load TTS provider based on environment variable"""
    provider_name = os.getenv("TTS_PROVIDER", "piper")

    logger.info(f"Loading TTS provider: {provider_name}")

    if provider_name == "piper":
        config = {
            "model": os.getenv("MODEL", "en_US-lessac-medium"),
            "device": os.getenv("DEVICE", "cpu"),
        }
        return PiperProvider(config)
    else:
        raise ValueError(f"Unknown TTS provider: {provider_name}")


async def serve():
    """Start gRPC server"""
    # Load provider
    provider = load_provider()

    # Initialize provider
    logger.info("Initializing TTS provider...")
    await provider.initialize()
    logger.info("TTS provider initialized successfully")

    # Create server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
        ],
    )

    # Add servicer
    tts_pb2_grpc.add_TTSServiceServicer_to_server(TTSServicer(provider), server)

    # Listen on port
    port = os.getenv("PORT", "50053")
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"Starting TTS service on port {port}...")
    await server.start()
    logger.info(f"TTS service running on port {port}")

    try:
        await server.wait_for_termination()
    finally:
        logger.info("Shutting down TTS service...")
        await provider.cleanup()
        await server.stop(grace=5)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    # Run server
    asyncio.run(serve())
