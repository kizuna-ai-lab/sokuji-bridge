#!/usr/bin/env python3
"""
STT Provider Comparison Script

Compare FasterWhisper vs SimulStreaming performance and latency.
"""

import asyncio
import time
import numpy as np
from loguru import logger

from src.providers.base import AudioChunk
from src.providers.stt.faster_whisper_provider import FasterWhisperProvider
from src.providers.stt.simulstreaming_provider import SimulStreamingProvider


async def generate_audio_stream(duration_s: float = 3.0, chunk_ms: float = 300):
    """Generate test audio stream (silence)"""
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    total_chunks = int(duration_s * 1000 / chunk_ms)

    for i in range(total_chunks):
        audio_data = np.zeros(chunk_samples, dtype=np.float32)
        audio_int16 = (audio_data * 32768).astype(np.int16)

        yield AudioChunk(
            data=audio_int16.tobytes(),
            sample_rate=sample_rate,
            timestamp=time.time(),
            channels=1,
            format="int16",
        )

        await asyncio.sleep(chunk_ms / 1000)  # Simulate real-time


async def benchmark_provider(provider_class, config, name: str):
    """Benchmark a specific STT provider"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {name}")
    logger.info(f"{'='*60}")

    provider = provider_class(config)

    # Initialize
    init_start = time.time()
    await provider.initialize()
    init_time = time.time() - init_start
    logger.info(f"✅ Initialization: {init_time:.2f}s")

    # Streaming test
    results = []
    stream_start = time.time()
    first_result_time = None

    logger.info("\n🎙️ Starting streaming transcription...")

    async for result in provider.transcribe_stream(generate_audio_stream()):
        result_time = time.time() - stream_start

        if first_result_time is None:
            first_result_time = result_time

        results.append({
            'time': result_time,
            'text': result.text,
            'is_final': result.is_final,
        })

        status = "✅ Final" if result.is_final else "📝 Progressive"
        logger.info(f"[{result_time:.3f}s] {status}: {result.text}")

    total_time = time.time() - stream_start

    # Statistics
    logger.info(f"\n📊 Statistics:")
    logger.info(f"  Total results: {len(results)}")
    logger.info(f"  First result latency: {first_result_time:.3f}s")
    logger.info(f"  Total processing time: {total_time:.2f}s")
    logger.info(f"  Progressive results: {sum(1 for r in results if not r['is_final'])}")
    logger.info(f"  Final results: {sum(1 for r in results if r['is_final'])}")

    await provider.cleanup()

    return {
        'name': name,
        'init_time': init_time,
        'first_result_latency': first_result_time,
        'total_time': total_time,
        'result_count': len(results),
        'progressive_count': sum(1 for r in results if not r['is_final']),
    }


async def main():
    """Run comparison"""
    logger.info("🚀 STT Provider Comparison")
    logger.info("="*60)

    # Test configurations
    faster_whisper_config = {
        "model_size": "medium",
        "device": "auto",
        "compute_type": "float16",
        "beam_size": 5,
    }

    simulstreaming_config = {
        "model_size": "large-v3",
        "device": "auto",
        "frame_threshold": 30,  # Low latency
        "beam_size": 5,
        "audio_min_len": 0.3,
        "vad_enabled": False,  # Disable for fair comparison
    }

    results = []

    # Benchmark FasterWhisper
    try:
        result = await benchmark_provider(
            FasterWhisperProvider,
            faster_whisper_config,
            "FasterWhisper (medium, beam=5)"
        )
        results.append(result)
    except Exception as e:
        logger.error(f"FasterWhisper benchmark failed: {e}")

    # Benchmark SimulStreaming
    try:
        result = await benchmark_provider(
            SimulStreamingProvider,
            simulstreaming_config,
            "SimulStreaming (large-v3, frame_threshold=30, beam=5)"
        )
        results.append(result)
    except Exception as e:
        logger.error(f"SimulStreaming benchmark failed: {e}")

    # Comparison Summary
    if len(results) == 2:
        logger.info(f"\n{'='*60}")
        logger.info("📊 Comparison Summary")
        logger.info(f"{'='*60}")

        logger.info(f"\n🏆 Winner Analysis:")

        # First result latency
        latencies = [(r['name'], r['first_result_latency']) for r in results]
        latencies.sort(key=lambda x: x[1])
        logger.info(f"\n⏱️  First Result Latency:")
        for i, (name, latency) in enumerate(latencies):
            emoji = "🥇" if i == 0 else "🥈"
            logger.info(f"  {emoji} {name}: {latency:.3f}s")

        improvement = (latencies[1][1] - latencies[0][1]) / latencies[1][1] * 100
        logger.info(f"  💡 Improvement: {improvement:.1f}% faster")

        # Progressive results
        logger.info(f"\n📝 Progressive Output:")
        for r in results:
            logger.info(f"  • {r['name']}: {r['progressive_count']} progressive results")

        # Total processing
        logger.info(f"\n⏰ Total Processing Time:")
        for r in results:
            logger.info(f"  • {r['name']}: {r['total_time']:.2f}s")

        logger.info(f"\n{'='*60}")
        logger.info("✨ Key Takeaways:")
        logger.info(f"{'='*60}")

        if latencies[0][0].startswith("SimulStreaming"):
            logger.info("✅ SimulStreaming provides significantly lower latency")
            logger.info("✅ Progressive results enable real-time feedback")
            logger.info("✅ AlignAtt policy successfully reduces waiting time")
        else:
            logger.info("⚠️  FasterWhisper was faster (unexpected)")
            logger.info("💡 Consider adjusting SimulStreaming frame_threshold")

    logger.info(f"\n{'='*60}")
    logger.info("✅ Comparison complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
