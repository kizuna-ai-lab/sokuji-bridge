#!/usr/bin/env python3
"""
Test script for SimulStreaming STT Provider

This script tests the basic functionality of the SimulStreamingProvider
including initialization, single chunk transcription, and streaming.
"""

import asyncio
import numpy as np
import time
from loguru import logger

# Setup logging
logger.add("simulstreaming_test.log", level="DEBUG")

from src.providers.base import AudioChunk
from src.providers.stt.simulstreaming_provider import SimulStreamingProvider


async def test_initialization():
    """Test provider initialization"""
    logger.info("=" * 60)
    logger.info("Test 1: Provider Initialization")
    logger.info("=" * 60)

    config = {
        "model_size": "small",  # small 模型仅需 ~2GB 显存
        "device": "auto",  # 自动检测 CUDA/CPU
        "compute_type": "int8",  # INT8 量化进一步降低显存 (~50%)
        "frame_threshold": 30,
        "beam_size": 1,  # Greedy 解码降低显存消耗
        "audio_min_len": 0.3,
        "min_chunk_size": 0.3,
        "vad_enabled": False,  # Disable VAD for simple test
        "audio_max_len": 20.0,  # 降低音频缓冲长度
    }

    provider = SimulStreamingProvider(config)

    try:
        logger.info("Initializing provider...")
        start = time.time()
        await provider.initialize()
        elapsed = time.time() - start

        logger.info(f"✅ Initialization successful in {elapsed:.2f}s")
        logger.info(f"Provider: {provider}")

        # Health check
        is_healthy = await provider.health_check()
        logger.info(f"Health check: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

        return provider

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise


async def test_single_transcription(provider: SimulStreamingProvider):
    """Test single audio chunk transcription"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Single Chunk Transcription")
    logger.info("=" * 60)

    # Generate 2 seconds of silence as test audio
    sample_rate = 16000
    duration_s = 2.0
    audio_data = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

    # Convert to int16 format
    audio_int16 = (audio_data * 32768).astype(np.int16)

    chunk = AudioChunk(
        data=audio_int16.tobytes(),
        sample_rate=sample_rate,
        timestamp=time.time(),
        channels=1,
        format="int16",
    )

    logger.info(f"Audio chunk: {chunk.duration_ms:.0f}ms, {len(chunk.data)} bytes")

    try:
        start = time.time()
        result = await provider.transcribe(chunk, language="zh")
        elapsed = time.time() - start

        logger.info(f"⏱️ Transcription took {elapsed:.3f}s")
        logger.info(f"📝 Result: {result}")
        logger.info(f"✅ Transcription successful")

        return result

    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        raise


async def test_streaming_transcription(provider: SimulStreamingProvider):
    """Test streaming audio transcription"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Streaming Transcription")
    logger.info("=" * 60)

    sample_rate = 16000
    chunk_duration_ms = 300  # 300ms chunks

    async def audio_generator():
        """Generate audio stream (silence for testing)"""
        for i in range(10):  # 10 chunks = 3 seconds total
            samples = int(sample_rate * chunk_duration_ms / 1000)
            audio_data = np.zeros(samples, dtype=np.float32)
            audio_int16 = (audio_data * 32768).astype(np.int16)

            chunk = AudioChunk(
                data=audio_int16.tobytes(),
                sample_rate=sample_rate,
                timestamp=time.time(),
                channels=1,
                format="int16",
            )

            logger.info(f"📦 Chunk {i+1}/10: {chunk.duration_ms:.0f}ms")
            yield chunk

            # Simulate real-time streaming
            await asyncio.sleep(chunk_duration_ms / 1000)

    try:
        start = time.time()
        result_count = 0

        logger.info("🎙️ Starting streaming transcription...")

        async for result in provider.transcribe_stream(audio_generator(), language="zh"):
            result_count += 1
            elapsed = time.time() - start

            logger.info(f"📝 Result #{result_count} at {elapsed:.3f}s:")
            logger.info(f"   Text: {result.text}")
            logger.info(f"   Language: {result.language}")
            logger.info(f"   Timing: {result.start_time:.2f}s - {result.end_time:.2f}s")
            logger.info(f"   Final: {result.is_final}")

        total_elapsed = time.time() - start
        logger.info(f"✅ Streaming complete: {result_count} results in {total_elapsed:.2f}s")

    except Exception as e:
        logger.error(f"❌ Streaming failed: {e}")
        raise


async def test_cleanup(provider: SimulStreamingProvider):
    """Test provider cleanup"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Provider Cleanup")
    logger.info("=" * 60)

    try:
        await provider.cleanup()
        logger.info("✅ Cleanup successful")

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        raise


async def main():
    """Run all tests"""
    logger.info("🚀 SimulStreaming STT Provider Test Suite")
    logger.info("=" * 60)

    try:
        # Test 1: Initialization
        provider = await test_initialization()

        # Test 2: Single transcription
        await test_single_transcription(provider)

        # Test 3: Streaming transcription
        await test_streaming_transcription(provider)

        # Test 4: Cleanup
        await test_cleanup(provider)

        logger.info("\n" + "=" * 60)
        logger.info("🎉 All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"❌ Test suite failed: {e}")
        logger.error(f"{'=' * 60}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
