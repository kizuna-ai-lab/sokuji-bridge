"""
Basic Usage Example for Sokuji-Bridge

Demonstrates how to use the translation pipeline with the fast local profile.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.manager import ConfigManager
from core.pipeline import TranslationPipeline
from providers.base import AudioChunk
from providers.stt.faster_whisper_provider import FasterWhisperProvider
from providers.translation.nllb_provider import NLLBProvider
from providers.tts.piper_provider import PiperProvider


async def test_single_translation():
    """Test single audio chunk translation"""
    print("=" * 60)
    print("Sokuji-Bridge - Basic Usage Example")
    print("=" * 60)
    print()

    # Load configuration
    print("📋 Loading configuration...")
    config_manager = ConfigManager.from_profile("fast")
    config = config_manager.get_config()
    print(f"✓ Loaded profile: {config.pipeline.name}")
    print(f"  Source: {config.pipeline.source_language}")
    print(f"  Target: {config.pipeline.target_language}")
    print()

    # Initialize providers
    print("🔧 Initializing providers...")

    # STT Provider
    stt_provider = FasterWhisperProvider(config.stt.config)
    print(f"  Initializing STT: {config.stt.provider}...")
    await stt_provider.initialize()
    print(f"  ✓ STT ready: {stt_provider}")

    # Translation Provider
    translation_provider = NLLBProvider(config.translation.config)
    print(f"  Initializing Translation: {config.translation.provider}...")
    await translation_provider.initialize()
    print(f"  ✓ Translation ready: {translation_provider}")

    # TTS Provider
    tts_provider = PiperProvider(config.tts.config)
    print(f"  Initializing TTS: {config.tts.provider}...")
    await tts_provider.initialize()
    print(f"  ✓ TTS ready: {tts_provider}")
    print()

    # Create pipeline
    print("🔗 Creating translation pipeline...")
    pipeline = TranslationPipeline(
        stt_provider=stt_provider,
        translation_provider=translation_provider,
        tts_provider=tts_provider,
        config=config,
    )
    await pipeline.initialize()
    print(f"✓ Pipeline ready: {pipeline.get_status().value}")
    print()

    # Test with dummy audio (1 second of generated audio)
    print("🎤 Creating test audio...")
    import numpy as np

    # Generate test audio: 1 second of speech-like noise
    sample_rate = 16000
    duration_s = 1.0
    audio_samples = np.random.randn(int(sample_rate * duration_s))
    audio_samples = (audio_samples * 0.1).astype(np.float32)  # Scale down
    audio_data = (audio_samples * 32768).astype(np.int16).tobytes()

    test_audio = AudioChunk(
        data=audio_data,
        sample_rate=sample_rate,
        timestamp=0.0,
        channels=1,
        format="int16",
    )
    print(f"✓ Created audio chunk: {test_audio.duration_ms:.0f}ms")
    print()

    # Process through pipeline
    print("⚡ Processing through pipeline...")
    print("-" * 60)
    try:
        result = await pipeline.process_single(test_audio)

        print("✅ Translation complete!")
        print(f"  Audio generated: {len(result.audio_data)} bytes")
        print(f"  Sample rate: {result.sample_rate}Hz")
        print(f"  Duration: {result.duration_ms:.0f}ms")
        print(f"  Voice: {result.voice_id}")
        print()

        # Show metrics
        metrics = pipeline.get_metrics()
        print("📊 Performance Metrics:")
        print(f"  STT Latency: {metrics['stt_latency_ms']:.1f}ms")
        print(f"  Translation Latency: {metrics['translation_latency_ms']:.1f}ms")
        print(f"  TTS Latency: {metrics['tts_latency_ms']:.1f}ms")
        print(f"  Total Latency: {metrics['total_latency_ms']:.1f}ms")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        await pipeline.cleanup()
        print("✓ Cleanup complete")
        print()

    print("=" * 60)
    print("Example completed successfully! 🎉")
    print("=" * 60)


async def test_provider_individually():
    """Test each provider individually"""
    print("=" * 60)
    print("Testing Individual Providers")
    print("=" * 60)
    print()

    # Test STT
    print("1️⃣  Testing STT Provider...")
    config_manager = ConfigManager.from_profile("fast")
    config = config_manager.get_config()

    stt = FasterWhisperProvider(config.stt.config)
    await stt.initialize()

    # Generate test audio
    import numpy as np
    audio_samples = np.random.randn(16000).astype(np.float32) * 0.1
    audio_data = (audio_samples * 32768).astype(np.int16).tobytes()
    test_audio = AudioChunk(audio_data, 16000, 0.0, 1, "int16")

    stt_result = await stt.transcribe(test_audio)
    print(f"  ✓ STT Result: {stt_result}")
    await stt.cleanup()
    print()

    # Test Translation
    print("2️⃣  Testing Translation Provider...")
    translator = NLLBProvider(config.translation.config)
    await translator.initialize()

    trans_result = await translator.translate(
        "Hello, how are you?",
        source_lang="en",
        target_lang="zh",
    )
    print(f"  ✓ Translation: {trans_result.original_text}")
    print(f"    → {trans_result.translated_text}")
    await translator.cleanup()
    print()

    # Test TTS
    print("3️⃣  Testing TTS Provider...")
    tts = PiperProvider(config.tts.config)
    await tts.initialize()

    tts_result = await tts.synthesize(
        "This is a test.",
        voice_id="default",
    )
    print(f"  ✓ TTS Result: {len(tts_result.audio_data)} bytes, {tts_result.duration_ms:.0f}ms")
    await tts.cleanup()
    print()

    print("=" * 60)
    print("All providers tested successfully! ✅")
    print("=" * 60)


def main():
    """Main entry point"""
    print()
    print("🌉 Sokuji-Bridge Example")
    print()

    try:
        # Run individual provider tests
        print("Running individual provider tests...\n")
        asyncio.run(test_provider_individually())

        print("\n" + "=" * 60 + "\n")

        # Run full pipeline test
        print("Running full pipeline test...\n")
        asyncio.run(test_single_translation())

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
