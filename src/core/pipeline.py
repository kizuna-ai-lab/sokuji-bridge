"""
Translation Pipeline Orchestrator

Coordinates STT, Translation, and TTS providers in a streaming pipeline.
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from providers.base import (
    STTProvider,
    TranslationProvider,
    TTSProvider,
    AudioChunk,
    TranscriptionResult,
    TranslationResult,
    SynthesisResult,
    ProviderStatus,
)
from config.schemas import SokujiBridgeConfig


class PipelineStatus(Enum):
    """Pipeline operational status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_audio_chunks: int = 0
    total_transcriptions: int = 0
    total_translations: int = 0
    total_syntheses: int = 0

    stt_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    errors: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_audio_chunks": self.total_audio_chunks,
            "total_transcriptions": self.total_transcriptions,
            "total_translations": self.total_translations,
            "total_syntheses": self.total_syntheses,
            "stt_latency_ms": self.stt_latency_ms,
            "translation_latency_ms": self.translation_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "errors": self.errors,
            "duration_s": (self.end_time - self.start_time) if self.start_time and self.end_time else 0.0,
        }


class TranslationPipeline:
    """
    Main translation pipeline orchestrator

    Coordinates STT → Translation → TTS in a streaming async pipeline.
    """

    def __init__(
        self,
        stt_provider: STTProvider,
        translation_provider: TranslationProvider,
        tts_provider: TTSProvider,
        config: SokujiBridgeConfig,
    ):
        """
        Initialize translation pipeline

        Args:
            stt_provider: Speech-to-text provider
            translation_provider: Translation provider
            tts_provider: Text-to-speech provider
            config: Pipeline configuration
        """
        self.stt = stt_provider
        self.translator = translation_provider
        self.tts = tts_provider
        self.config = config

        self.status = PipelineStatus.IDLE
        self.metrics = PipelineMetrics()

        # Async queues for pipeline stages
        self._audio_queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue(maxsize=10)
        self._text_queue: asyncio.Queue[Optional[TranscriptionResult]] = asyncio.Queue(maxsize=10)
        self._translated_queue: asyncio.Queue[Optional[TranslationResult]] = asyncio.Queue(maxsize=10)
        self._audio_out_queue: asyncio.Queue[Optional[SynthesisResult]] = asyncio.Queue(maxsize=10)

        # Control flags
        self._stop_event = asyncio.Event()
        self._workers: list[asyncio.Task] = []

    async def initialize(self) -> None:
        """
        Initialize all providers

        Raises:
            RuntimeError: If initialization fails
        """
        self.status = PipelineStatus.INITIALIZING

        try:
            # Initialize providers in parallel
            await asyncio.gather(
                self.stt.initialize(),
                self.translator.initialize(),
                self.tts.initialize(),
            )

            # Verify all providers are ready
            if not await self.stt.health_check():
                raise RuntimeError("STT provider not healthy after initialization")
            if not await self.translator.health_check():
                raise RuntimeError("Translation provider not healthy after initialization")
            if not await self.tts.health_check():
                raise RuntimeError("TTS provider not healthy after initialization")

            self.status = PipelineStatus.IDLE

        except Exception as e:
            self.status = PipelineStatus.ERROR
            raise RuntimeError(f"Pipeline initialization failed: {e}") from e

    async def cleanup(self) -> None:
        """Clean up all providers"""
        self.status = PipelineStatus.STOPPED

        # Stop all workers
        self._stop_event.set()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

        # Cleanup providers in parallel
        await asyncio.gather(
            self.stt.cleanup(),
            self.translator.cleanup(),
            self.tts.cleanup(),
            return_exceptions=True,
        )

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SynthesisResult]:
        """
        Process audio stream through the complete pipeline

        Args:
            audio_stream: Input audio stream

        Yields:
            Synthesized audio results

        Example:
            >>> async for audio_chunk in input_stream:
            >>>     async for result in pipeline.process_audio_stream([audio_chunk]):
            >>>         play_audio(result.audio_data)
        """
        if self.status != PipelineStatus.IDLE:
            raise RuntimeError(f"Pipeline not ready (status: {self.status})")

        self.status = PipelineStatus.RUNNING
        self.metrics.start_time = time.time()
        self._stop_event.clear()

        try:
            # Start worker coroutines
            self._workers = [
                asyncio.create_task(self._stt_worker()),
                asyncio.create_task(self._translation_worker()),
                asyncio.create_task(self._tts_worker()),
            ]

            # Feed audio stream into pipeline
            audio_feeder = asyncio.create_task(self._feed_audio_stream(audio_stream))

            # Yield results as they become available
            async for result in self._consume_output():
                yield result

            # Wait for audio feeder to complete
            await audio_feeder

        finally:
            # Signal workers to stop
            self._stop_event.set()

            # Send termination signals
            await self._audio_queue.put(None)
            await self._text_queue.put(None)
            await self._translated_queue.put(None)

            # Wait for all workers to finish
            if self._workers:
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers.clear()

            self.metrics.end_time = time.time()
            self.status = PipelineStatus.IDLE

    async def process_single(
        self,
        audio: AudioChunk,
    ) -> SynthesisResult:
        """
        Process a single audio chunk (non-streaming)

        Args:
            audio: Audio chunk to process

        Returns:
            Synthesized audio result
        """
        start_time = time.time()

        try:
            # STT
            stt_start = time.time()
            transcription = await self.stt.transcribe(
                audio,
                language=self.config.pipeline.source_language
                if self.config.pipeline.source_language != "auto"
                else None,
            )
            stt_duration = (time.time() - stt_start) * 1000
            self.metrics.stt_latency_ms = stt_duration
            self.metrics.total_transcriptions += 1

            # Translation
            trans_start = time.time()
            translation = await self.translator.translate(
                transcription.text,
                source_lang=transcription.language,
                target_lang=self.config.pipeline.target_language,
            )
            trans_duration = (time.time() - trans_start) * 1000
            self.metrics.translation_latency_ms = trans_duration
            self.metrics.total_translations += 1

            # TTS
            tts_start = time.time()
            voice_id = self.config.tts.voice or "default"
            synthesis = await self.tts.synthesize(
                translation.translated_text,
                voice_id=voice_id,
                language=self.config.pipeline.target_language,
            )
            tts_duration = (time.time() - tts_start) * 1000
            self.metrics.tts_latency_ms = tts_duration
            self.metrics.total_syntheses += 1

            # Total latency
            total_duration = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms = total_duration

            return synthesis

        except Exception as e:
            self.metrics.errors += 1
            raise RuntimeError(f"Pipeline processing failed: {e}") from e

    async def _feed_audio_stream(self, audio_stream: AsyncIterator[AudioChunk]) -> None:
        """Feed audio stream into pipeline"""
        try:
            async for audio_chunk in audio_stream:
                if self._stop_event.is_set():
                    break
                await self._audio_queue.put(audio_chunk)
                self.metrics.total_audio_chunks += 1
        finally:
            await self._audio_queue.put(None)  # Signal end of stream

    async def _stt_worker(self) -> None:
        """STT worker coroutine"""
        try:
            while not self._stop_event.is_set():
                audio = await self._audio_queue.get()
                if audio is None:  # End of stream signal
                    break

                try:
                    start_time = time.time()
                    result = await self.stt.transcribe(
                        audio,
                        language=self.config.pipeline.source_language
                        if self.config.pipeline.source_language != "auto"
                        else None,
                    )
                    duration_ms = (time.time() - start_time) * 1000

                    self.metrics.stt_latency_ms = (
                        self.metrics.stt_latency_ms * 0.9 + duration_ms * 0.1
                    )  # Exponential moving average
                    self.metrics.total_transcriptions += 1

                    await self._text_queue.put(result)

                except Exception as e:
                    self.metrics.errors += 1
                    # Log error but continue processing
                    print(f"STT error: {e}")

        finally:
            await self._text_queue.put(None)  # Signal end of stream

    async def _translation_worker(self) -> None:
        """Translation worker coroutine"""
        try:
            while not self._stop_event.is_set():
                transcription = await self._text_queue.get()
                if transcription is None:  # End of stream signal
                    break

                try:
                    start_time = time.time()
                    result = await self.translator.translate(
                        transcription.text,
                        source_lang=transcription.language,
                        target_lang=self.config.pipeline.target_language,
                    )
                    duration_ms = (time.time() - start_time) * 1000

                    self.metrics.translation_latency_ms = (
                        self.metrics.translation_latency_ms * 0.9 + duration_ms * 0.1
                    )
                    self.metrics.total_translations += 1

                    await self._translated_queue.put(result)

                except Exception as e:
                    self.metrics.errors += 1
                    print(f"Translation error: {e}")

        finally:
            await self._translated_queue.put(None)  # Signal end of stream

    async def _tts_worker(self) -> None:
        """TTS worker coroutine"""
        try:
            while not self._stop_event.is_set():
                translation = await self._translated_queue.get()
                if translation is None:  # End of stream signal
                    break

                try:
                    start_time = time.time()
                    voice_id = self.config.tts.voice or "default"
                    result = await self.tts.synthesize(
                        translation.translated_text,
                        voice_id=voice_id,
                        language=translation.target_language,
                    )
                    duration_ms = (time.time() - start_time) * 1000

                    self.metrics.tts_latency_ms = (
                        self.metrics.tts_latency_ms * 0.9 + duration_ms * 0.1
                    )
                    self.metrics.total_syntheses += 1

                    await self._audio_out_queue.put(result)

                except Exception as e:
                    self.metrics.errors += 1
                    print(f"TTS error: {e}")

        finally:
            await self._audio_out_queue.put(None)  # Signal end of stream

    async def _consume_output(self) -> AsyncIterator[SynthesisResult]:
        """Consume output from pipeline"""
        while True:
            result = await self._audio_out_queue.get()
            if result is None:  # End of stream signal
                break
            yield result

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.metrics.to_dict()

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status"""
        return self.status

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check

        Returns:
            Dictionary with health status of all components
        """
        return {
            "pipeline_status": self.status.value,
            "stt_healthy": await self.stt.health_check(),
            "translation_healthy": await self.translator.health_check(),
            "tts_healthy": await self.tts.health_check(),
            "metrics": self.get_metrics(),
        }

    def __repr__(self) -> str:
        return (
            f"TranslationPipeline(status={self.status.value}, "
            f"stt={self.stt.__class__.__name__}, "
            f"translator={self.translator.__class__.__name__}, "
            f"tts={self.tts.__class__.__name__})"
        )
