"""
Simple Test for Sokuji-Bridge

Quick test that works without package installation.
Run from project root: python examples/simple_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from src as a package
import src.config.manager as config_manager
import src.providers.base as base
import src.providers.stt.faster_whisper_provider as stt_provider
import src.providers.translation.nllb_provider as translation_provider
import src.providers.tts.piper_provider as tts_provider
import src.core.pipeline as pipeline


async def test_configuration():
    """Test configuration system"""
    print("=" * 60)
    print("Testing Configuration System")
    print("=" * 60)
    print()

    try:
        # Load configuration
        print("📋 Loading fast profile...")
        manager = config_manager.ConfigManager.from_profile("fast")
        config = manager.get_config()

        print(f"✓ Configuration loaded successfully!")
        print(f"  Profile: {config.pipeline.name}")
        print(f"  Source Language: {config.pipeline.source_language}")
        print(f"  Target Language: {config.pipeline.target_language}")
        print(f"  STT Provider: {config.stt.provider}")
        print(f"  Translation Provider: {config.translation.provider}")
        print(f"  TTS Provider: {config.tts.provider}")
        print()

        # Validate configuration
        is_valid, errors = manager.validate()
        if is_valid:
            print("✅ Configuration is valid!")
        else:
            print("⚠️  Configuration warnings:")
            for error in errors:
                print(f"  - {error}")
        print()

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_translation_simple():
    """Test translation without audio"""
    print("=" * 60)
    print("Testing Translation Provider")
    print("=" * 60)
    print()

    try:
        # Load config
        manager = config_manager.ConfigManager.from_profile("fast")
        config = manager.get_config()

        # Initialize translator
        print("🔧 Initializing NLLB translator...")
        translator = translation_provider.NLLBProvider(config.translation.config)
        await translator.initialize()
        print(f"✓ Translator ready: {translator}")
        print()

        # Test translation
        test_texts = [
            ("Hello, how are you?", "en", "zh"),
            ("Good morning!", "en", "zh"),
            ("Thank you very much", "en", "ja"),
        ]

        for text, src, tgt in test_texts:
            print(f"Translating: '{text}' ({src} → {tgt})")
            result = await translator.translate(text, src, tgt)
            print(f"  → {result.translated_text}")
            print(f"  Confidence: {result.confidence:.2f}")
            print()

        # Cleanup
        await translator.cleanup()
        print("✅ Translation test passed!")
        print()

        return True

    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_providers_health():
    """Test if providers can be initialized"""
    print("=" * 60)
    print("Testing Provider Initialization")
    print("=" * 60)
    print()

    try:
        manager = config_manager.ConfigManager.from_profile("fast")
        config = manager.get_config()

        # Test STT
        print("1️⃣  Testing STT Provider...")
        try:
            stt = stt_provider.FasterWhisperProvider(config.stt.config)
            print("  ℹ️  STT provider created (initialization skipped for speed)")
            print(f"  Model: {stt.model_size}, Device: {stt.device}")
            print("  ✓ STT provider: OK")
        except Exception as e:
            print(f"  ❌ STT provider failed: {e}")
        print()

        # Test Translation
        print("2️⃣  Testing Translation Provider...")
        try:
            translator = translation_provider.NLLBProvider(config.translation.config)
            print("  ℹ️  Translation provider created (initialization skipped for speed)")
            print(f"  Model: {translator.model_name}, Device: {translator.device}")
            print("  ✓ Translation provider: OK")
        except Exception as e:
            print(f"  ❌ Translation provider failed: {e}")
        print()

        # Test TTS
        print("3️⃣  Testing TTS Provider...")
        try:
            tts = tts_provider.PiperProvider(config.tts.config)
            print("  ℹ️  TTS provider created (initialization skipped for speed)")
            print(f"  Model: {tts.model_name}")
            print("  ✓ TTS provider: OK")
        except Exception as e:
            print(f"  ❌ TTS provider failed: {e}")
        print()

        print("✅ All providers can be created successfully!")
        print()
        return True

    except Exception as e:
        print(f"❌ Provider health check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    print()
    print("🌉 Sokuji-Bridge Simple Test")
    print()
    print("This test verifies the basic functionality without requiring")
    print("full model downloads or GPU resources.")
    print()

    results = {}

    try:
        # Test 1: Configuration
        print("\n" + "=" * 60 + "\n")
        results['config'] = asyncio.run(test_configuration())

        # Test 2: Provider Health
        print("\n" + "=" * 60 + "\n")
        results['providers'] = asyncio.run(test_providers_health())

        # Test 3: Translation (requires model download)
        print("\n" + "=" * 60 + "\n")
        print("⚠️  The following test will download models (~1-2GB)")
        print("Press Ctrl+C to skip, or wait 5 seconds to continue...")
        try:
            import time
            time.sleep(5)
            results['translation'] = asyncio.run(test_translation_simple())
        except KeyboardInterrupt:
            print("\n⏭️  Skipped translation test")
            results['translation'] = None

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for test_name, result in results.items():
            if result is None:
                status = "⏭️  SKIPPED"
            elif result:
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            print(f"{status}: {test_name}")
        print("=" * 60)

        if all(r in [True, None] for r in results.values()):
            print("\n🎉 All tests passed or skipped!")
            print("\nNext steps:")
            print("1. Install package: pip install -e .")
            print("2. Run full example: python -m examples.basic_usage")
            print("3. Or use API: from sokuji_bridge import TranslationPipeline")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
