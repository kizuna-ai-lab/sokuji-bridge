#!/usr/bin/env python3
"""
SimulStreaming Real-Time Microphone Transcription

Comprehensive example for real-time speech transcription with optional VAD.

Features:
- Real-time microphone audio capture
- Low-latency streaming transcription with SimulStreaming (<2s)
- Optional VAD for automatic sentence segmentation
- Real-time display of transcription results
- Graceful shutdown with Ctrl+C

Usage:
    # Basic mode (without VAD)
    python examples/microphone_simulstreaming.py

    # With VAD automatic segmentation
    python examples/microphone_simulstreaming.py --vad

    # With custom VAD parameters
    python examples/microphone_simulstreaming.py --vad --silence 300 --threshold 0.6

Display:
    Without VAD:
        📝 Progressive text updates (same line, real-time)
        ✅ Final result shown when stopped (Ctrl+C)

    With VAD:
        📝 In-progress text (real-time)
        🎯 Complete sentence (VAD detected silence)

Configuration:
    --vad                    Enable VAD automatic segmentation
    --no-vad                 Disable VAD (default)
    --model SIZE             Model size: tiny, small, medium, large, large-v3 (default: small)
    --language LANG          Language code: zh, en, ja, etc. (default: zh)
    --silence MS             VAD silence duration in milliseconds (default: 500)
    --threshold FLOAT        VAD detection threshold 0.0-1.0 (default: 0.5)
    --device DEVICE          Device: cuda, cpu, auto (default: auto)
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.microphone import MicrophoneInput
from src.providers.stt.simulstreaming_provider import SimulStreamingProvider


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SimulStreaming Real-Time Microphone Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--vad",
        action="store_true",
        default=False,
        help="Enable VAD for automatic sentence segmentation",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (default behavior)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large", "large-v2", "large-v3"],
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code for transcription (default: zh)",
    )
    parser.add_argument(
        "--silence",
        type=int,
        default=500,
        help="VAD silence duration in ms to trigger sentence end (default: 500)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD speech detection threshold 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device for inference (default: auto)",
    )

    args = parser.parse_args()

    # Handle --no-vad flag
    if args.no_vad:
        args.vad = False

    return args


async def main():
    """Main transcription loop"""
    args = parse_args()

    # Display configuration
    print(f"\n🎙️  SimulStreaming Real-Time Transcription")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Device: {args.device}")
    print(f"VAD: {'✅ Enabled' if args.vad else '❌ Disabled'}")
    if args.vad:
        print(f"  - Silence threshold: {args.silence}ms")
        print(f"  - Detection threshold: {args.threshold}")
    print(f"{'='*70}\n")

    # Initialize SimulStreaming provider
    print("⏳ Loading model...")
    config = {
        "model_size": args.model,
        "device": args.device,
        "frame_threshold": 40,  # Balanced latency
        "language": args.language,
    }

    # Add VAD configuration if enabled
    if args.vad:
        config.update({
            "vad_enabled": True,
            "vad_threshold": args.threshold,
            "vad_min_silence_ms": args.silence,
            "vad_speech_pad_ms": 100,
            "vad_min_buffered_length": 1.0,
            "vac_chunk_size": 0.04,
        })

    provider = SimulStreamingProvider(config)
    await provider.initialize()
    print("✅ Model ready\n")

    # Initialize microphone
    print("🎤 Initializing microphone...")
    mic = MicrophoneInput(sample_rate=16000, channels=1)
    await mic.start()
    print("✅ Microphone ready\n")

    # Display instructions
    if args.vad:
        print("🎙️  Listening with VAD... (pause 0.5s between sentences)\n")
        print("💬 Tip: VAD will automatically detect sentence boundaries\n")
    else:
        print("🎙️  Listening... (Press Ctrl+C to stop)\n")
        print("💬 Tip: Use --vad flag for automatic sentence segmentation\n")

    print("-" * 70)
    print()

    sentence_count = 0
    last_text = ""

    try:
        # Stream audio and get transcription
        async def audio_stream():
            async for chunk in mic.stream():
                yield chunk

        async for result in provider.transcribe_stream(audio_stream()):
            if result.is_final:
                # Sentence completed (VAD detected or stream ended)
                if args.vad:
                    sentence_count += 1
                    print(f"\r🎯 [Sentence {sentence_count}] {result.text}")
                else:
                    print(f"\r✅ {result.text}")
                print()  # New line
                last_text = ""
            else:
                # In-progress text
                print(f"\r📝 {result.text}", end="", flush=True)
                last_text = result.text

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping recording")
        if last_text:
            if args.vad:
                print(f"🎯 [Incomplete] {last_text}\n")
                print(f"📊 Statistics: Recognized {sentence_count} complete sentences\n")
            else:
                print(f"✅ {last_text} (partial result at interruption)\n")
        else:
            print()

    finally:
        # Cleanup
        await mic.stop()
        await provider.cleanup()
        print("✅ Cleanup complete\n")


if __name__ == "__main__":
    asyncio.run(main())
