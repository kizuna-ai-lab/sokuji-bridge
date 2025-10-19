"""
Minimal STT Test - Microphone → VAD → STT Only

This script tests ONLY the speech recognition part of the pipeline.
Use this to diagnose if the issue is in STT or elsewhere.
"""

import asyncio
import sys
from pathlib import Path
import time

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.manager import ConfigManager
from providers.stt.faster_whisper_provider import FasterWhisperProvider
from utils.microphone import MicrophoneInput
from utils.vad import EnergyVAD


async def test_stt_only(
    source_lang: str = "zh",
    profile: str = "fast",
    use_vad: bool = True,
    vad_energy_threshold: float = 0.005,
    min_speech_duration_ms: float = 100,
    min_silence_duration_ms: float = 300,
    chunk_duration_ms: float = 2000,
):
    """
    Test microphone → VAD → STT pipeline only

    Args:
        source_lang: Source language code
        profile: Configuration profile
        use_vad: Use VAD for segmentation (if False, use fixed-duration chunks)
        vad_energy_threshold: VAD energy threshold
        min_speech_duration_ms: Minimum speech duration
        min_silence_duration_ms: Minimum silence duration
        chunk_duration_ms: Duration of fixed chunks when VAD is disabled
    """
    print()
    print("=" * 70)
    print("🧪 STT-Only Test - Microphone → VAD → STT")
    print("=" * 70)
    print("This test verifies if speech recognition is working correctly.")
    print()

    # 1. Initialize microphone
    print("🎤 Initializing microphone...")
    mic = MicrophoneInput(sample_rate=16000, channels=1)
    await mic.start()

    mic_device = mic.get_device_info()
    if mic_device:
        print(f"✓ Microphone: [{mic_device.index}] {mic_device.name}")
    else:
        print(f"✓ Microphone: Default device")
    print()

    # 2. Initialize VAD (optional)
    vad = None
    if use_vad:
        print("🎙️  Initializing VAD...")
        vad = EnergyVAD(
            energy_threshold=vad_energy_threshold,
            sample_rate=16000
        )
        await vad.initialize()
        print(f"✓ VAD ready (threshold={vad_energy_threshold}, "
              f"min_speech={min_speech_duration_ms}ms, "
              f"min_silence={min_silence_duration_ms}ms)")
    else:
        print(f"⚠️  VAD disabled - using fixed {chunk_duration_ms}ms chunks")
    print()

    # 3. Load configuration and initialize STT
    print("📋 Loading configuration...")
    config_manager = ConfigManager.from_profile(profile)
    config = config_manager.get_config()
    config.pipeline.source_language = source_lang
    print(f"✓ Profile: {config.pipeline.name}")
    print(f"  STT Provider: {config.stt.provider}")
    print()

    print("🔧 Initializing STT provider...")
    stt_provider = FasterWhisperProvider(config.stt.config)
    print(f"  Loading STT model...")
    await stt_provider.initialize()
    print(f"✓ STT provider initialized")
    print()

    # 4. Start listening and transcribing
    print("=" * 70)
    print("🎤 Listening for speech... (Press Ctrl+C to stop)")
    print(f"   Speak in {source_lang} (Chinese)")
    if use_vad:
        print(f"   Mode: VAD-based segmentation")
    else:
        print(f"   Mode: Fixed {chunk_duration_ms}ms chunks")
    print("=" * 70)
    print()

    try:
        segment_count = 0

        if use_vad:
            # Use VAD to segment audio
            audio_stream = vad.segment_audio(
                mic.stream(),
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        else:
            # Manual chunking without VAD
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

        # Process each speech segment
        async for audio_segment in audio_stream:
            segment_count += 1

            # Calculate segment info
            import numpy as np
            audio_array = np.frombuffer(audio_segment.data, dtype=np.int16)
            duration_ms = (len(audio_array) / audio_segment.sample_rate) * 1000

            print(f"📦 [Segment #{segment_count}]")
            print(f"   Received audio segment: {len(audio_segment.data)} bytes, {duration_ms:.1f}ms")
            print(f"   Sample rate: {audio_segment.sample_rate}Hz, Channels: {audio_segment.channels}")
            print(f"   Format: {audio_segment.format}")
            print()

            # Try to transcribe
            print(f"   🔄 Calling STT provider...")
            try:
                stt_start = time.time()

                result = await stt_provider.transcribe(
                    audio_segment,
                    language=source_lang if source_lang != "auto" else None,
                )

                stt_duration = (time.time() - stt_start) * 1000

                print(f"   ✓ STT completed in {stt_duration:.0f}ms")
                print()
                print(f"   📝 Recognized Text ({result.language}): {result.text}")
                print(f"      Confidence: {result.confidence:.2f}")
                print(f"      Is Final: {result.is_final}")

            except Exception as e:
                print(f"   ❌ STT Error: {e}")
                import traceback
                traceback.print_exc()

            print()
            print("-" * 70)
            print()

    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print()
        print("🧹 Cleaning up...")
        await mic.stop()
        if vad:
            await vad.cleanup()
        await stt_provider.cleanup()
        print("✓ Cleanup complete")

    print()
    print("=" * 70)
    print(f"✅ STT test complete! Processed {segment_count} segments.")
    print("=" * 70)


def main():
    """Main entry point"""
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║              🧪 STT-Only Test Tool                                 ║")
    print("║                                                                    ║")
    print("║  This tool tests ONLY the speech recognition (STT) component      ║")
    print("║  to help diagnose issues with the full translation pipeline.      ║")
    print("║                                                                    ║")
    print("║  Pipeline: Microphone → VAD → STT                                 ║")
    print("║  (Skips translation and TTS for faster debugging)                 ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Disable VAD for faster testing - use fixed 2-second chunks
        asyncio.run(
            test_stt_only(
                source_lang="zh",
                profile="fast",
                use_vad=False,  # Disable VAD to avoid waiting for silence
                chunk_duration_ms=2000,  # Process every 2 seconds
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
