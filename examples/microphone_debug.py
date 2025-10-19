"""
Microphone Debug Tool

Real-time visualization of microphone input and VAD detection.
Use this to diagnose VAD issues and find optimal threshold values.
"""

import asyncio
import sys
from pathlib import Path
import time

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.microphone import MicrophoneInput
from utils.vad import EnergyVAD
from providers.base import AudioChunk


async def debug_microphone_with_vad(
    vad_energy_threshold: float = 0.005,
    min_speech_duration_ms: float = 100,
    min_silence_duration_ms: float = 300,
    duration_seconds: int = 30,
):
    """
    Debug microphone input with real-time VAD visualization

    Args:
        vad_energy_threshold: VAD energy threshold
        min_speech_duration_ms: Minimum speech duration
        min_silence_duration_ms: Minimum silence duration
        duration_seconds: How long to run the debug session
    """
    print()
    print("=" * 70)
    print("🔧 Microphone + VAD Debug Tool")
    print("=" * 70)
    print()
    print("This tool shows real-time audio energy levels and VAD detection")
    print("to help you find the optimal VAD parameters for your microphone.")
    print()

    # Initialize microphone
    print("🎤 Initializing microphone...")
    mic = MicrophoneInput(sample_rate=16000, channels=1)
    await mic.start()

    mic_device = mic.get_device_info()
    if mic_device:
        print(f"✓ Microphone: [{mic_device.index}] {mic_device.name}")
    else:
        print(f"✓ Microphone: Default device")
    print()

    # Initialize VAD
    print("🎙️  Initializing VAD...")
    vad = EnergyVAD(
        energy_threshold=vad_energy_threshold,
        sample_rate=16000
    )
    await vad.initialize()
    print(f"✓ VAD ready")
    print(f"  Energy threshold: {vad_energy_threshold}")
    print(f"  Min speech duration: {min_speech_duration_ms}ms")
    print(f"  Min silence duration: {min_silence_duration_ms}ms")
    print()

    # Start monitoring
    print("=" * 70)
    print(f"📊 Monitoring for {duration_seconds} seconds... (Press Ctrl+C to stop)")
    print()
    print("Legend:")
    print("  🔇 Silence (energy < threshold)")
    print("  🗣️  Speech detected (energy >= threshold)")
    print("  📦 Speech segment emitted (met duration requirements)")
    print()
    print("=" * 70)
    print()

    try:
        chunk_count = 0
        speech_count = 0
        silence_count = 0
        segment_count = 0
        start_time = time.time()

        # Create audio stream with VAD segmentation
        audio_stream = vad.segment_audio(
            mic.stream(),
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )

        # Create a raw audio stream for per-chunk monitoring
        raw_stream = mic.stream()

        # Run both streams concurrently
        async def monitor_raw_chunks():
            """Monitor individual audio chunks"""
            nonlocal chunk_count, speech_count, silence_count

            timeout_time = start_time + duration_seconds

            async for chunk in raw_stream:
                if time.time() > timeout_time:
                    break

                chunk_count += 1

                # Detect speech in this chunk
                result = await vad.detect_speech(chunk)

                # Calculate energy for display
                import numpy as np
                audio_array = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(audio_array ** 2))

                # Create energy bar
                bar_length = 50
                energy_bar = "█" * int(energy * 1000) if energy > 0 else ""
                energy_bar = energy_bar[:bar_length].ljust(bar_length, "░")

                # Status indicator
                if result.is_speech:
                    status = "🗣️  SPEECH"
                    speech_count += 1
                else:
                    status = "🔇 silence"
                    silence_count += 1

                # Print chunk info
                print(f"[{chunk_count:4d}] {status} | Energy: {energy:.4f} | {energy_bar}")

        async def monitor_segments():
            """Monitor emitted speech segments"""
            nonlocal segment_count

            timeout_time = start_time + duration_seconds

            async for segment in audio_stream:
                if time.time() > timeout_time:
                    break

                segment_count += 1

                # Calculate segment duration
                import numpy as np
                audio_array = np.frombuffer(segment.data, dtype=np.int16)
                duration_ms = (len(audio_array) / segment.sample_rate) * 1000

                print()
                print(f"📦 SEGMENT #{segment_count} EMITTED")
                print(f"   Duration: {duration_ms:.1f}ms")
                print(f"   Size: {len(segment.data)} bytes")
                print()

        # Run both monitoring tasks
        await asyncio.gather(
            monitor_raw_chunks(),
            monitor_segments(),
        )

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
        await vad.cleanup()
        print("✓ Cleanup complete")

    # Show statistics
    elapsed_time = time.time() - start_time
    print()
    print("=" * 70)
    print("📊 Session Statistics")
    print("=" * 70)
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Total chunks: {chunk_count}")
    print(f"Speech chunks: {speech_count} ({speech_count/chunk_count*100:.1f}%)")
    print(f"Silence chunks: {silence_count} ({silence_count/chunk_count*100:.1f}%)")
    print(f"Segments emitted: {segment_count}")
    print()

    # Recommendations
    print("💡 Recommendations:")
    if segment_count == 0:
        print("  ⚠️  No segments emitted!")
        print("  Try:")
        print(f"    - Lower energy threshold (current: {vad_energy_threshold})")
        print(f"    - Reduce min_speech_duration_ms (current: {min_speech_duration_ms}ms)")
        print(f"    - Reduce min_silence_duration_ms (current: {min_silence_duration_ms}ms)")
    elif segment_count < 3:
        print("  ⚠️  Very few segments emitted")
        print("  Consider lowering VAD thresholds for faster response")
    else:
        print("  ✓ VAD parameters look good!")
        print(f"    Emitted {segment_count} segments in {elapsed_time:.1f}s")
    print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug microphone and VAD settings"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="VAD energy threshold (default: 0.005)",
    )
    parser.add_argument(
        "--min-speech",
        type=float,
        default=100,
        help="Minimum speech duration in ms (default: 100)",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=300,
        help="Minimum silence duration in ms (default: 300)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="How long to run debug session in seconds (default: 30)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            debug_microphone_with_vad(
                vad_energy_threshold=args.threshold,
                min_speech_duration_ms=args.min_speech,
                min_silence_duration_ms=args.min_silence,
                duration_seconds=args.duration,
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
