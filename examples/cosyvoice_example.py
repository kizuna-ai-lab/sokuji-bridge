#!/usr/bin/env python3
"""
CosyVoice2 TTS Provider Examples

Demonstrates cross-lingual synthesis for translation scenarios.
"""

import asyncio
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from providers.tts.cosyvoice_provider import CosyVoiceProvider


async def example_cross_lingual_translation():
    """
    Example 1: Cross-lingual Translation (CORE USE CASE)

    Scenario: Translate Chinese speech to English while preserving voice
    """
    print("\n" + "="*70)
    print("Example 1: Cross-lingual Translation")
    print("="*70)

    config = {
        "model": "CosyVoice2-0.5B",
        "model_dir": "pretrained_models/CosyVoice2-0.5B",
        "inference_mode": "cross_lingual",
        "device": "cuda",  # Change to "cpu" if no GPU
        "streaming": {"enabled": True},
        "cross_lingual": {
            "auto_extract_prompt": True,
            "prompt_duration_sec": 3.0,
        },
    }

    provider = CosyVoiceProvider(config)

    print("\n📥 Initializing CosyVoice2 model...")
    await provider.initialize()
    print(f"✓ Model loaded: {provider.model_name} (v{provider.model_version})")

    # Simulate source audio (Chinese speech)
    # In real scenario, this would come from STT output
    print("\n🎤 Simulating Chinese speech audio (5 seconds)...")
    source_audio = np.random.randn(16000 * 5).astype(np.float32)

    # Translated text (from translation service)
    target_text = "Hello, how are you today? I hope you're having a wonderful day!"
    print(f"\n📝 Target text (English): '{target_text}'")

    # Synthesize: Speak English with Chinese voice
    print("\n🔊 Synthesizing English speech with Chinese voice...")
    result = await provider.synthesize(
        text=target_text,
        voice_id="cross_lingual",
        prompt_audio=source_audio,
        stream=True,
    )

    print(f"✓ Generated {result.duration_ms:.0f}ms of audio")
    print(f"  Sample rate: {result.sample_rate}Hz")
    print(f"  Audio size: {len(result.audio_data)} bytes")

    await provider.cleanup()
    print("\n✓ Cross-lingual translation complete!")


async def example_voice_cloning():
    """
    Example 2: Zero-shot Voice Cloning

    Scenario: Clone any voice from a short sample
    """
    print("\n" + "="*70)
    print("Example 2: Zero-shot Voice Cloning")
    print("="*70)

    config = {
        "model": "CosyVoice2-0.5B",
        "model_dir": "pretrained_models/CosyVoice2-0.5B",
        "inference_mode": "zero_shot",
        "device": "cuda",
        "zero_shot": {
            "speaker_embedding_cache": True,
        },
    }

    provider = CosyVoiceProvider(config)

    print("\n📥 Initializing CosyVoice2 model...")
    await provider.initialize()

    # Reference audio and transcript
    reference_audio = np.random.randn(16000 * 5).astype(np.float32)
    reference_text = "This is a sample of my voice speaking naturally."

    print(f"\n📝 Reference text: '{reference_text}'")

    # Synthesize new text in cloned voice
    new_text = "Now I'm speaking completely different words, but in the same voice!"
    print(f"📝 New text: '{new_text}'")

    print("\n🔊 Cloning voice and synthesizing new text...")
    result = await provider.synthesize(
        text=new_text,
        voice_id="zero_shot",
        prompt_audio=reference_audio,
        prompt_text=reference_text,
    )

    print(f"✓ Generated {result.duration_ms:.0f}ms of cloned audio")

    await provider.cleanup()
    print("\n✓ Voice cloning complete!")


async def example_streaming_synthesis():
    """
    Example 3: Streaming Synthesis

    Scenario: Process text stream with low latency
    """
    print("\n" + "="*70)
    print("Example 3: Streaming Synthesis")
    print("="*70)

    config = {
        "model": "CosyVoice2-0.5B",
        "model_dir": "pretrained_models/CosyVoice2-0.5B",
        "inference_mode": "cross_lingual",
        "device": "cuda",
        "streaming": {
            "enabled": True,
            "token_hop_len": 50,
        },
    }

    provider = CosyVoiceProvider(config)

    print("\n📥 Initializing CosyVoice2 model...")
    await provider.initialize()

    # Text stream generator
    async def text_stream():
        chunks = [
            "First sentence of the speech. ",
            "Second sentence with more content. ",
            "Third and final sentence here.",
        ]
        for i, chunk in enumerate(chunks, 1):
            print(f"  📄 Chunk {i}: {chunk}")
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    reference_audio = np.random.randn(16000 * 3).astype(np.float32)

    print("\n🔊 Streaming synthesis...")
    total_duration = 0
    chunk_count = 0

    async for result in provider.synthesize_stream(
        text_stream(),
        voice_id="cross_lingual",
        prompt_audio=reference_audio,
    ):
        chunk_count += 1
        total_duration += result.duration_ms
        print(f"  🎵 Got audio chunk {chunk_count}: {result.duration_ms:.0f}ms")

    print(f"\n✓ Streaming complete: {chunk_count} chunks, {total_duration:.0f}ms total")

    await provider.cleanup()


async def example_list_voices():
    """
    Example 4: List Available Voices

    Scenario: Discover available voices and modes
    """
    print("\n" + "="*70)
    print("Example 4: List Available Voices")
    print("="*70)

    config = {
        "model": "CosyVoice2-0.5B",
        "model_dir": "pretrained_models/CosyVoice2-0.5B",
        "device": "cuda",
    }

    provider = CosyVoiceProvider(config)

    print("\n📥 Initializing CosyVoice2 model...")
    await provider.initialize()

    print("\n📋 Available voices and modes:")
    voices = await provider.get_voices()

    # Group by mode
    modes = {}
    for voice in voices:
        mode = voice["mode"]
        if mode not in modes:
            modes[mode] = []
        modes[mode].append(voice)

    for mode, voice_list in modes.items():
        print(f"\n  {mode.upper()} Mode:")
        for voice in voice_list:
            print(f"    - {voice['id']}: {voice['name']}")
            if voice.get('description'):
                print(f"      {voice['description']}")

    await provider.cleanup()


async def example_full_translation_pipeline():
    """
    Example 5: Complete Translation Pipeline

    Scenario: End-to-end translation with voice preservation
    """
    print("\n" + "="*70)
    print("Example 5: Full Translation Pipeline")
    print("="*70)

    print("\n🔄 Translation Pipeline:")
    print("  STT (Chinese) → Translation → TTS (English with Chinese voice)")

    config = {
        "model": "CosyVoice2-0.5B",
        "model_dir": "pretrained_models/CosyVoice2-0.5B",
        "inference_mode": "cross_lingual",
        "device": "cuda",
        "streaming": {"enabled": True},
    }

    provider = CosyVoiceProvider(config)
    await provider.initialize()

    # Step 1: STT (simulated)
    print("\n1️⃣ STT: Transcribing source audio...")
    source_audio = np.random.randn(16000 * 4).astype(np.float32)
    source_text = "你好，今天天气怎么样？"  # Chinese
    source_lang = "zh"
    print(f"   Source text: '{source_text}' (language: {source_lang})")

    # Step 2: Translation (simulated)
    print("\n2️⃣ Translation: Chinese → English...")
    target_text = "Hello, how is the weather today?"
    target_lang = "en"
    print(f"   Target text: '{target_text}' (language: {target_lang})")

    # Step 3: TTS with voice preservation
    print("\n3️⃣ TTS: Synthesizing English with Chinese voice...")
    result = await provider.synthesize(
        text=target_text,
        voice_id="cross_lingual",
        prompt_audio=source_audio,  # Use original Chinese audio!
        stream=True,
    )

    print(f"   ✓ Generated {result.duration_ms:.0f}ms of audio")
    print(f"   Result: English text in Chinese speaker's voice")

    await provider.cleanup()

    print("\n✓ Full pipeline complete!")
    print("  The output audio speaks English text with the Chinese speaker's voice")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("CosyVoice2 TTS Provider Examples")
    print("="*70)

    try:
        # Run examples
        await example_cross_lingual_translation()
        await example_voice_cloning()
        await example_streaming_synthesis()
        await example_list_voices()
        await example_full_translation_pipeline()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure CosyVoice model is downloaded")
        print("  2. Check CUDA availability if using GPU")
        print("  3. Verify all dependencies are installed")
        print("  4. See docs/cosyvoice_provider.md for detailed setup")


if __name__ == "__main__":
    asyncio.run(main())
