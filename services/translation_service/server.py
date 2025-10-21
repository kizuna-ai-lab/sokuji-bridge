"""
Translation Service - gRPC Server

Translation microservice that wraps translation providers.
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

from src.generated import translation_pb2, translation_pb2_grpc, common_pb2
from src.providers.base import TranslationProvider, TranslationResult as ProviderTranslationResult
from src.providers.translation.nllb_provider import NLLBProvider


def translation_to_proto(result: ProviderTranslationResult) -> common_pb2.TranslationResult:
    """Convert provider TranslationResult to proto"""
    return common_pb2.TranslationResult(
        original_text=result.original_text,
        translated_text=result.translated_text,
        source_language=result.source_language,
        target_language=result.target_language,
        confidence=result.confidence,
        timestamp=result.timestamp,
        model_name=result.model_name or "",
    )


class TranslationServicer(translation_pb2_grpc.TranslationServiceServicer):
    """Translation Service implementation"""

    def __init__(self, provider: TranslationProvider):
        self.provider = provider
        logger.info(f"Translation Service initialized with provider: {provider.__class__.__name__}")

    async def Translate(
        self,
        request: translation_pb2.TranslateRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.TranslationResult:
        """Translate a single text"""
        try:
            logger.debug(f"Translate request: {request.source_language} -> {request.target_language}")

            # Translate
            result = await self.provider.translate(
                text=request.text,
                source_lang=request.source_language,
                target_lang=request.target_language,
                context=request.context if request.context else None,
            )

            # Convert to proto
            return translation_to_proto(result)

        except Exception as e:
            logger.error(f"Translation error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Translation failed: {str(e)}")

    async def TranslateBatch(
        self,
        request: translation_pb2.TranslateBatchRequest,
        context: grpc.aio.ServicerContext,
    ) -> translation_pb2.TranslateBatchResponse:
        """Translate multiple texts in batch"""
        try:
            logger.debug(f"TranslateBatch request: {len(request.texts)} texts")

            # Batch translate
            results = await self.provider.translate_batch(
                texts=list(request.texts),
                source_lang=request.source_language,
                target_lang=request.target_language,
            )

            # Convert to proto
            proto_results = [translation_to_proto(r) for r in results]
            return translation_pb2.TranslateBatchResponse(results=proto_results)

        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Batch translation failed: {str(e)}")

    async def TranslateStream(
        self,
        request_iterator: AsyncIterator[translation_pb2.TranslateRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.TranslationResult]:
        """Translate text stream"""
        try:
            logger.debug("TranslateStream started")

            # Extract first request to get language pair
            first_request = await request_iterator.__anext__()
            source_lang = first_request.source_language
            target_lang = first_request.target_language

            # Create text stream
            async def text_stream():
                yield first_request.text
                async for req in request_iterator:
                    yield req.text

            # Stream translation
            async for result in self.provider.translate_stream(
                text_stream(), source_lang, target_lang
            ):
                yield translation_to_proto(result)

        except Exception as e:
            logger.error(f"Streaming translation error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Streaming translation failed: {str(e)}")

    async def GetSupportedLanguages(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> translation_pb2.LanguageListResponse:
        """Get supported languages"""
        try:
            languages = self.provider.supported_languages()
            return translation_pb2.LanguageListResponse(
                language_codes=languages,
                supports_batch=self.provider.supports_batch(),
                supports_streaming=self.provider.supports_streaming(),
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
                    "supports_batch": str(self.provider.supports_batch()),
                    "supports_streaming": str(self.provider.supports_streaming()),
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


def load_provider() -> TranslationProvider:
    """Load translation provider based on environment variable"""
    provider_name = os.getenv("TRANSLATION_PROVIDER", "nllb_local")

    logger.info(f"Loading translation provider: {provider_name}")

    if provider_name == "nllb_local":
        config = {
            "model": os.getenv("MODEL", "facebook/nllb-200-distilled-1.3B"),
            "device": os.getenv("DEVICE", "cuda"),
            "precision": os.getenv("PRECISION", "float16"),
        }
        return NLLBProvider(config)
    else:
        raise ValueError(f"Unknown translation provider: {provider_name}")


async def serve():
    """Start gRPC server"""
    # Load provider
    provider = load_provider()

    # Initialize provider
    logger.info("Initializing translation provider...")
    await provider.initialize()
    logger.info("Translation provider initialized successfully")

    # Create server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ],
    )

    # Add servicer
    translation_pb2_grpc.add_TranslationServiceServicer_to_server(
        TranslationServicer(provider), server
    )

    # Listen on port
    port = os.getenv("PORT", "50052")
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"Starting translation service on port {port}...")
    await server.start()
    logger.info(f"Translation service running on port {port}")

    try:
        await server.wait_for_termination()
    finally:
        logger.info("Shutting down translation service...")
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
