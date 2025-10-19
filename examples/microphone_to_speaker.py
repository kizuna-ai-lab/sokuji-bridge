"""
Real-time Microphone Translation Example

Demonstrates real-time translation from microphone to speaker with VAD.
Captures audio from microphone, translates it, and plays the result through speakers.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.manager import ConfigManager
from core.pipeline import TranslationPipeline
from providers.stt.faster_whisper_provider import FasterWhisperProvider
from providers.translation.nllb_provider import NLLBProvider
from providers.tts.piper_provider import PiperProvider
from utils.microphone import MicrophoneInput
from utils.speaker import SpeakerOutput
from utils.vad import EnergyVAD


async def real_time_translation(
    source_lang: str = "zh",
    target_lang: str = "en",
    config_path: Optional[Path] = None,
    use_vad: bool = False,
    vad_energy_threshold: float = 0.005,
    min_speech_duration_ms: float = 100,
    min_silence_duration_ms: float = 300,
    chunk_duration_ms: float = 2000,
):
    """
    Real-time microphone to speaker translation

    Args:
        source_lang: Source language code (default: zh - Chinese)
        target_lang: Target language code (default: en - English)
        config_path: Path to configuration file (default: configs/default.yaml)
        use_vad: Use VAD for segmentation (default: False - use fixed chunks)
        vad_energy_threshold: VAD energy threshold for speech detection (default: 0.005)
        min_speech_duration_ms: Minimum speech duration in ms (default: 100)
        min_silence_duration_ms: Minimum silence duration to split segments (default: 300)
        chunk_duration_ms: Fixed chunk duration when VAD is disabled (default: 2000ms)
    """
    print("=" * 70)
    print("🌉 Sokuji-Bridge - Real-time Translation Demo")
    print("=" * 70)
    print(f"Translation: {source_lang} → {target_lang}")
    print()

    # 1. Initialize audio I/O devices
    print("🎤 Initializing microphone...")
    mic = MicrophoneInput(sample_rate=16000, channels=1)
    await mic.start()

    mic_device = mic.get_device_info()
    if mic_device:
        print(f"✓ Microphone: [{mic_device.index}] {mic_device.name}")
    else:
        print(f"✓ Microphone: Default device")
    print()

    print("🔊 Initializing speaker...")
    speaker = SpeakerOutput(sample_rate=16000, channels=1)
    await speaker.start()

    speaker_device = speaker.get_device_info()
    if speaker_device:
        print(f"✓ Speaker: [{speaker_device.index}] {speaker_device.name}")
    else:
        print(f"✓ Speaker: Default device")
    print()

    # 2. Load configuration
    print("📋 Loading configuration...")
    config_manager = ConfigManager(config_path=config_path)
    config = config_manager.get_config()

    # Override language settings
    config.pipeline.source_language = source_lang
    config.pipeline.target_language = target_lang

    print(f"✓ Configuration: {config.pipeline.name}")
    print(f"  STT: {config.stt.provider}")
    print(f"  Translation: {config.translation.provider}")
    print(f"  TTS: {config.tts.provider}")
    print()

    # 3. Initialize providers
    print("🔧 Initializing translation providers...")

    stt_provider = FasterWhisperProvider(config.stt.config)
    print(f"  Initializing STT...")
    await stt_provider.initialize()

    translation_provider = NLLBProvider(config.translation.config)
    print(f"  Initializing Translation...")
    await translation_provider.initialize()

    tts_provider = PiperProvider(config.tts.config)
    print(f"  Initializing TTS...")
    await tts_provider.initialize()

    print("✓ All providers ready")
    print()

    # 4. Create translation pipeline
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

    # 5. Initialize VAD (optional)
    vad = None
    if use_vad:
        print("🎙️  Initializing VAD for intelligent speech segmentation...")
        vad = EnergyVAD(
            energy_threshold=vad_energy_threshold,
            sample_rate=16000
        )
        await vad.initialize()
        print(f"✓ VAD ready: threshold={vad_energy_threshold}, "
              f"min_speech={min_speech_duration_ms}ms, "
              f"min_silence={min_silence_duration_ms}ms")
    else:
        print(f"⚠️  VAD disabled - using fixed {chunk_duration_ms}ms chunks for faster response")
    print()

    # 6. Start real-time translation
    print("=" * 70)
    print("🎤 Listening... (Press Ctrl+C to stop)")
    print("   Speak in Chinese to hear English translation")
    if use_vad:
        print(f"   Mode: VAD-based segmentation (stop speaking for {min_silence_duration_ms}ms to trigger)")
    else:
        print(f"   Mode: Fixed {chunk_duration_ms}ms chunks (processes automatically)")
    print("=" * 70)
    print()

    try:
        speech_count = 0

        if use_vad:
            # Use VAD to segment audio at natural speech boundaries
            audio_stream = vad.segment_audio(
                mic.stream(),
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        else:
            # Use fixed-duration chunks
            async def fixed_duration_chunker():
                """Accumulate audio chunks into fixed-duration segments"""
                import numpy as np
                from providers.base import AudioChunk

                buffer = []
                buffer_duration_ms = 0.0

                async for chunk in mic.stream():
                    buffer.append(chunk.data)

                    # Calculate duration of current chunk
                    audio_array = np.frombuffer(chunk.data, dtype=np.int16)
                    chunk_dur_ms = (len(audio_array) / chunk.sample_rate) * 1000
                    buffer_duration_ms += chunk_dur_ms

                    # When we have enough audio, yield it
                    if buffer_duration_ms >= chunk_duration_ms:
                        combined_audio = b''.join(buffer)
                        yield AudioChunk(
                            data=combined_audio,
                            sample_rate=chunk.sample_rate,
                            timestamp=chunk.timestamp,
                            channels=chunk.channels,
                            format=chunk.format,
                        )

                        # Reset buffer
                        buffer.clear()
                        buffer_duration_ms = 0.0

            audio_stream = fixed_duration_chunker()

        # Stream through pipeline and play results
        async for result in pipeline.process_audio_stream(audio_stream):
            # Skip empty recognitions (silence) to improve real-time performance
            if not result.transcription or not result.transcription.text.strip():
                continue

            speech_count += 1

            print(f"[Speech #{speech_count}]")

            # Print STT result (speech recognition)
            if result.transcription:
                print(f"  📝 STT ({result.transcription.language}): {result.transcription.text}")
                print(f"     Confidence: {result.transcription.confidence:.2f}")

            # Print Translation result
            if result.translation:
                print(f"  🌐 Translation ({result.translation.source_language}→{result.translation.target_language}): "
                      f"{result.translation.translated_text}")
                print(f"     Confidence: {result.translation.confidence:.2f}")

            # Print audio info
            print(f"  🔊 Playing translated audio...")
            print(f"  ⏱️  Duration: {result.duration_ms:.0f}ms")
            print()

            # Play translated audio through speaker
            await speaker.play(result)

    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
    except Exception as e:
        print()
        print(f"❌ Error during translation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup all resources
        print()
        print("🧹 Cleaning up...")
        await mic.stop()
        await speaker.stop()
        if vad:
            await vad.cleanup()
        await pipeline.cleanup()
        print("✓ Cleanup complete")

    # Show final metrics
    print()
    print("📊 Translation Metrics:")
    metrics = pipeline.get_metrics()
    print(f"  Total speech segments processed: {speech_count}")
    print(f"  Total audio chunks: {metrics['total_audio_chunks']}")
    print(f"  Transcriptions: {metrics['total_transcriptions']}")
    print(f"  Translations: {metrics['total_translations']}")
    print(f"  Syntheses: {metrics['total_syntheses']}")
    print()
    print(f"  Average STT latency: {metrics['stt_latency_ms']:.1f}ms")
    print(f"  Average translation latency: {metrics['translation_latency_ms']:.1f}ms")
    print(f"  Average TTS latency: {metrics['tts_latency_ms']:.1f}ms")
    print(f"  Total pipeline latency: {metrics['total_latency_ms']:.1f}ms")
    print()

    print("=" * 70)
    print("✅ Real-time translation session complete!")
    print("=" * 70)


def main():
    """Main entry point"""
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║           🎤 Real-time Microphone Translation 🔊                    ║")
    print("║                                                                    ║")
    print("║  This demo captures audio from your microphone, translates it      ║")
    print("║  in real-time, and plays the translated audio through speakers.    ║")
    print("║                                                                    ║")
    print("║  Features:                                                         ║")
    print("║  • Voice Activity Detection (VAD) for intelligent segmentation     ║")
    print("║  • Real-time Speech-to-Text (STT)                                  ║")
    print("║  • Neural Machine Translation                                      ║")
    print("║  • Text-to-Speech (TTS) synthesis                                  ║")
    print("║                                                                    ║")
    print("║  Default: Chinese (zh) → English (en)                              ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Run with hardcoded Chinese → English translation
        # Using fixed-duration chunks for reliable real-time performance
        asyncio.run(
            real_time_translation(
                source_lang="zh",
                target_lang="en",
                config_path=None,  # Use default configs/default.yaml
                use_vad=False,  # Disable VAD - use fixed chunks instead
                chunk_duration_ms=2000,  # Process every 2 seconds automatically
            )
        )

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
