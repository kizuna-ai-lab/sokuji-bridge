"""
Audio File Translation Example

Demonstrates translating audio files and saving the translated output.
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
from utils.audio_file import AudioFileReader, AudioFileWriter


async def translate_audio_file(
    input_file: str,
    output_file: str,
    source_lang: str = "zh",
    target_lang: str = "en",
    config_path: Optional[Path] = None,
    use_vad: bool = True,
    vad_energy_threshold: float = 0.01,
    min_speech_duration_ms: float = 250,
    min_silence_duration_ms: float = 500,
):
    """
    Translate audio file and save result

    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        source_lang: Source language code
        target_lang: Target language code
        config_path: Path to configuration file (default: configs/default.yaml)
        use_vad: Use VAD for intelligent segmentation (default: True)
        vad_energy_threshold: VAD energy threshold (default: 0.01)
        min_speech_duration_ms: Minimum speech duration in ms (default: 250)
        min_silence_duration_ms: Minimum silence duration in ms (default: 500)
    """
    print("=" * 70)
    print("🌉 Sokuji-Bridge - File Translation Demo")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Source → Target: {source_lang} → {target_lang}")
    print()

    # 1. Initialize audio file I/O
    print("📁 Opening input file...")
    # Use smaller chunks (100ms) for VAD analysis, or larger for fixed-duration
    chunk_duration = 100 if use_vad else 1000
    reader = AudioFileReader(input_file, chunk_duration_ms=chunk_duration, target_sample_rate=16000)
    await reader.start()
    file_info = reader.get_info()
    print(f"✓ File opened: {file_info['duration_ms']:.1f}ms, "
          f"{file_info['sample_rate']}Hz, {file_info['channels']}ch")
    print()

    print("📁 Creating output file...")
    writer = AudioFileWriter(output_file, sample_rate=16000, channels=1)
    await writer.start()
    print(f"✓ Output file ready: {output_file}")
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

    # 4. Create pipeline
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

    # 4.5. Initialize VAD if enabled
    vad = None
    if use_vad:
        print("🎙️  Initializing VAD for intelligent segmentation...")
        vad = EnergyVAD(
            energy_threshold=vad_energy_threshold,
            sample_rate=16000
        )
        await vad.initialize()
        print(f"✓ VAD ready: threshold={vad_energy_threshold}, "
              f"min_speech={min_speech_duration_ms}ms, "
              f"min_silence={min_silence_duration_ms}ms")
        print()

    # 5. Process file through pipeline
    print("⚡ Processing audio file...")
    if use_vad:
        print("   Using VAD for intelligent segmentation at natural pauses")
    else:
        print(f"   Using fixed {chunk_duration}ms chunks")
    print("=" * 70)

    try:
        chunk_count = 0
        total_duration_ms = 0.0

        # Prepare audio stream (with or without VAD)
        if use_vad:
            # VAD segments audio at natural speech boundaries
            audio_stream = vad.segment_audio(
                reader.stream(),
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
        else:
            # Direct stream with fixed-duration chunks
            audio_stream = reader.stream()

        # Stream through pipeline and write to output file
        async for result in pipeline.process_audio_stream(audio_stream):
            chunk_count += 1
            total_duration_ms += result.duration_ms

            print(f"\n[Chunk {chunk_count}]")

            # Print STT result (speech recognition)
            if result.transcription:
                print(f"  📝 STT ({result.transcription.language}): {result.transcription.text}")
                print(f"     Confidence: {result.transcription.confidence:.2f}")

            # Print Translation result
            if result.translation:
                print(f"  🌐 Translation ({result.translation.source_language}→{result.translation.target_language}): "
                      f"{result.translation.translated_text}")
                print(f"     Confidence: {result.translation.confidence:.2f}")

            # Print processing time
            print(f"  ⏱️  Duration: {result.duration_ms:.0f}ms | Total: {total_duration_ms:.1f}ms")

            # Write to output file
            await writer.play(result)

        print("=" * 70)
        print(f"✓ Processed {chunk_count} chunks")
        print()

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        await writer.stop()
        await reader.stop()
        if vad:
            await vad.cleanup()
        await pipeline.cleanup()
        print("✓ Cleanup complete")

    # Show final metrics
    print()
    print("📊 Translation Metrics:")
    metrics = pipeline.get_metrics()
    print(f"  Input duration: {file_info['duration_ms']:.1f}ms")
    print(f"  Output duration: {writer.get_info()['duration_ms']:.1f}ms")
    print(f"  Chunks processed: {metrics['total_audio_chunks']}")
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
    print(f"✅ Translation complete! Output saved to: {output_file}")
    print("=" * 70)


async def batch_translate_files(
    input_files: list[str],
    output_dir: str,
    source_lang: str = "zh",
    target_lang: str = "en",
    use_vad: bool = True,
    vad_energy_threshold: float = 0.01,
    min_speech_duration_ms: float = 250,
    min_silence_duration_ms: float = 500,
):
    """
    Batch translate multiple audio files

    Args:
        input_files: List of input file paths
        output_dir: Output directory path
        source_lang: Source language code
        target_lang: Target language code
        profile: Configuration profile to use
        use_vad: Use VAD for intelligent segmentation
        vad_energy_threshold: VAD energy threshold
        min_speech_duration_ms: Minimum speech duration in ms
        min_silence_duration_ms: Minimum silence duration in ms
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🌉 Batch translating {len(input_files)} files...")
    print(f"Output directory: {output_dir}")
    print()

    for i, input_file in enumerate(input_files, 1):
        input_path = Path(input_file)
        output_file = output_path / f"{input_path.stem}_translated.wav"

        print(f"\n{'=' * 70}")
        print(f"File {i}/{len(input_files)}: {input_path.name}")
        print(f"{'=' * 70}\n")

        try:
            await translate_audio_file(
                input_file=str(input_file),
                output_file=str(output_file),
                source_lang=source_lang,
                target_lang=target_lang,
                config_path=None,  # Use default configs/default.yaml
                use_vad=use_vad,
                vad_energy_threshold=vad_energy_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )
        except Exception as e:
            print(f"❌ Failed to translate {input_file}: {e}")
            continue

    print(f"\n✅ Batch translation complete! Processed {len(input_files)} files.")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate audio files using Sokuji-Bridge"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Input audio file(s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file or directory (for batch mode)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="zh",
        help="Source language code (default: zh)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="en",
        help="Target language code (default: en)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD, use fixed-duration chunks (default: VAD enabled)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.01,
        help="VAD energy threshold for speech detection (default: 0.01)",
    )
    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=250,
        help="Minimum speech duration in ms (default: 250)",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=500,
        help="Minimum silence duration in ms to split segments (default: 500)",
    )

    args = parser.parse_args()

    try:
        # Batch mode if multiple input files
        if len(args.input) > 1:
            output_dir = args.output or "translated_outputs"
            asyncio.run(
                batch_translate_files(
                    input_files=args.input,
                    output_dir=output_dir,
                    source_lang=args.source,
                    target_lang=args.target,
                    use_vad=not args.no_vad,
                    vad_energy_threshold=args.vad_threshold,
                    min_speech_duration_ms=args.min_speech_duration,
                    min_silence_duration_ms=args.min_silence_duration,
                )
            )
        else:
            # Single file mode
            input_file = args.input[0]
            input_path = Path(input_file)

            if args.output:
                output_file = args.output
            else:
                output_file = f"{input_path.stem}_translated.wav"

            asyncio.run(
                translate_audio_file(
                    input_file=input_file,
                    output_file=output_file,
                    source_lang=args.source,
                    target_lang=args.target,
                    config_path=None,  # Use default configs/default.yaml
                    use_vad=not args.no_vad,
                    vad_energy_threshold=args.vad_threshold,
                    min_speech_duration_ms=args.min_speech_duration,
                    min_silence_duration_ms=args.min_silence_duration,
                )
            )

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
