"""
STT Service - gRPC Server

Speech-to-Text microservice that wraps STT providers.
"""

import asyncio
import os
import sys
import time
from concurrent import futures
from typing import AsyncIterator, Optional

import grpc
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.generated import stt_pb2, stt_pb2_grpc, common_pb2
from src.providers.base import STTProvider, AudioChunk as ProviderAudioChunk, TranscriptionResult as ProviderTranscriptionResult
from src.providers.stt.faster_whisper_provider import FasterWhisperProvider


def audio_chunk_from_proto(proto: common_pb2.AudioChunk) -> ProviderAudioChunk:
    """Convert proto AudioChunk to provider AudioChunk"""
    return ProviderAudioChunk(
        data=proto.data,
        sample_rate=proto.sample_rate,
        timestamp=proto.timestamp,
        channels=proto.channels,
        format=proto.format,
    )


def transcription_to_proto(result: ProviderTranscriptionResult) -> common_pb2.TranscriptionResult:
    """Convert provider TranscriptionResult to proto"""
    return common_pb2.TranscriptionResult(
        text=result.text,
        language=result.language,
        confidence=result.confidence,
        timestamp=result.timestamp,
        start_time=result.start_time,
        end_time=result.end_time,
        is_final=result.is_final,
    )


class STTServicer(stt_pb2_grpc.STTServiceServicer):
    """STT Service implementation"""

    def __init__(self, provider: STTProvider):
        self.provider = provider
        logger.info(f"STT Service initialized with provider: {provider.__class__.__name__}")

    async def Transcribe(
        self,
        request: stt_pb2.TranscribeRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.TranscriptionResult:
        """Transcribe a single audio chunk"""
        try:
            logger.debug(f"Transcribe request received, audio size: {len(request.audio.data)} bytes")

            # Convert proto to provider format
            audio = audio_chunk_from_proto(request.audio)
            language = request.language if request.language else None

            # Transcribe
            result = await self.provider.transcribe(audio, language=language)

            # Convert back to proto
            return transcription_to_proto(result)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Transcription failed: {str(e)}")

    async def TranscribeStream(
        self,
        request_iterator: AsyncIterator[common_pb2.AudioChunk],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.TranscriptionResult]:
        """Transcribe audio stream"""
        try:
            logger.debug("TranscribeStream started")

            # Convert proto stream to provider format
            async def audio_stream():
                async for audio_proto in request_iterator:
                    yield audio_chunk_from_proto(audio_proto)

            # Stream transcription
            async for result in self.provider.transcribe_stream(audio_stream()):
                yield transcription_to_proto(result)

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Streaming transcription failed: {str(e)}")

    async def GetSupportedLanguages(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> stt_pb2.LanguageListResponse:
        """Get supported languages"""
        try:
            languages = self.provider.supported_languages()
            return stt_pb2.LanguageListResponse(language_codes=languages)
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
                details={"supports_streaming": str(self.provider.supports_streaming())},
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


def load_provider() -> STTProvider:
    """Load STT provider based on environment variable"""
    provider_name = os.getenv("STT_PROVIDER", "faster_whisper")

    logger.info(f"Loading STT provider: {provider_name}")

    if provider_name == "faster_whisper":
        config = {
            "model_size": os.getenv("MODEL_SIZE", "medium"),
            "device": os.getenv("DEVICE", "cuda"),
            "compute_type": os.getenv("COMPUTE_TYPE", "float16"),
            "vad_filter": os.getenv("VAD_FILTER", "true").lower() == "true",
        }
        return FasterWhisperProvider(config)
    else:
        raise ValueError(f"Unknown STT provider: {provider_name}")


async def serve():
    """Start gRPC server"""
    # Load provider
    provider = load_provider()

    # Initialize provider
    logger.info("Initializing STT provider...")
    await provider.initialize()
    logger.info("STT provider initialized successfully")

    # Create server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
        ],
    )

    # Add servicer
    stt_pb2_grpc.add_STTServiceServicer_to_server(STTServicer(provider), server)

    # Listen on port
    port = os.getenv("PORT", "50051")
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"Starting STT service on port {port}...")
    await server.start()
    logger.info(f"STT service running on port {port}")

    try:
        await server.wait_for_termination()
    finally:
        logger.info("Shutting down STT service...")
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
