"""
gRPC Clients for Gateway Service

Manages connections to STT, Translation, and TTS services.
"""

import asyncio
import os
import sys
from typing import Optional, AsyncIterator

import grpc
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.generated import (
    stt_pb2,
    stt_pb2_grpc,
    translation_pb2,
    translation_pb2_grpc,
    tts_pb2,
    tts_pb2_grpc,
    common_pb2,
)


class STTClient:
    """gRPC client for STT service"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[stt_pb2_grpc.STTServiceStub] = None
        logger.info(f"STT client initialized for {service_url}")

    async def connect(self):
        """Connect to STT service"""
        self.channel = grpc.aio.insecure_channel(
            self.service_url,
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = stt_pb2_grpc.STTServiceStub(self.channel)
        logger.info(f"Connected to STT service at {self.service_url}")

    async def disconnect(self):
        """Disconnect from STT service"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from STT service")

    async def transcribe(
        self, audio: common_pb2.AudioChunk, language: Optional[str] = None
    ) -> common_pb2.TranscriptionResult:
        """Transcribe audio chunk"""
        request = stt_pb2.TranscribeRequest(audio=audio, language=language)
        return await self.stub.Transcribe(request)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[common_pb2.AudioChunk]
    ) -> AsyncIterator[common_pb2.TranscriptionResult]:
        """Transcribe audio stream"""
        async for result in self.stub.TranscribeStream(audio_stream):
            yield result

    async def health_check(self) -> common_pb2.HealthCheckResponse:
        """Check service health"""
        return await self.stub.HealthCheck(common_pb2.Empty())


class TranslationClient:
    """gRPC client for Translation service"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[translation_pb2_grpc.TranslationServiceStub] = None
        logger.info(f"Translation client initialized for {service_url}")

    async def connect(self):
        """Connect to Translation service"""
        self.channel = grpc.aio.insecure_channel(
            self.service_url,
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        self.stub = translation_pb2_grpc.TranslationServiceStub(self.channel)
        logger.info(f"Connected to Translation service at {self.service_url}")

    async def disconnect(self):
        """Disconnect from Translation service"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from Translation service")

    async def translate(
        self, text: str, source_lang: str, target_lang: str, context: Optional[str] = None
    ) -> common_pb2.TranslationResult:
        """Translate text"""
        request = translation_pb2.TranslateRequest(
            text=text,
            source_language=source_lang,
            target_language=target_lang,
            context=context,
        )
        return await self.stub.Translate(request)

    async def translate_stream(
        self,
        text_stream: AsyncIterator[str],
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[common_pb2.TranslationResult]:
        """Translate text stream"""

        async def request_stream():
            async for text in text_stream:
                yield translation_pb2.TranslateRequest(
                    text=text,
                    source_language=source_lang,
                    target_language=target_lang,
                )

        async for result in self.stub.TranslateStream(request_stream()):
            yield result

    async def health_check(self) -> common_pb2.HealthCheckResponse:
        """Check service health"""
        return await self.stub.HealthCheck(common_pb2.Empty())


class TTSClient:
    """gRPC client for TTS service"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[tts_pb2_grpc.TTSServiceStub] = None
        logger.info(f"TTS client initialized for {service_url}")

    async def connect(self):
        """Connect to TTS service"""
        self.channel = grpc.aio.insecure_channel(
            self.service_url,
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = tts_pb2_grpc.TTSServiceStub(self.channel)
        logger.info(f"Connected to TTS service at {self.service_url}")

    async def disconnect(self):
        """Disconnect from TTS service"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from TTS service")

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> common_pb2.SynthesisResult:
        """Synthesize speech"""
        request = tts_pb2.SynthesizeRequest(
            text=text,
            voice_id=voice_id,
            language=language,
            parameters=kwargs,
        )
        return await self.stub.Synthesize(request)

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        voice_id: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[common_pb2.SynthesisResult]:
        """Synthesize speech from text stream"""

        async def request_stream():
            async for text in text_stream:
                yield tts_pb2.SynthesizeRequest(
                    text=text,
                    voice_id=voice_id,
                    language=language,
                    parameters=kwargs,
                )

        async for result in self.stub.SynthesizeStream(request_stream()):
            yield result

    async def health_check(self) -> common_pb2.HealthCheckResponse:
        """Check service health"""
        return await self.stub.HealthCheck(common_pb2.Empty())


class GRPCClients:
    """Manages all gRPC clients"""

    def __init__(
        self,
        stt_url: str = "localhost:50051",
        translation_url: str = "localhost:50052",
        tts_url: str = "localhost:50053",
    ):
        self.stt = STTClient(stt_url)
        self.translation = TranslationClient(translation_url)
        self.tts = TTSClient(tts_url)

    async def connect_all(self):
        """Connect to all services"""
        logger.info("Connecting to all gRPC services...")
        await asyncio.gather(
            self.stt.connect(),
            self.translation.connect(),
            self.tts.connect(),
        )
        logger.info("All gRPC services connected")

    async def disconnect_all(self):
        """Disconnect from all services"""
        logger.info("Disconnecting from all gRPC services...")
        await asyncio.gather(
            self.stt.disconnect(),
            self.translation.disconnect(),
            self.tts.disconnect(),
        )
        logger.info("All gRPC services disconnected")

    async def health_check_all(self) -> dict:
        """Check health of all services"""
        results = await asyncio.gather(
            self.stt.health_check(),
            self.translation.health_check(),
            self.tts.health_check(),
            return_exceptions=True,
        )

        return {
            "stt": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "translation": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "tts": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
        }
