# CosyVoice2 TTS Provider Guide

## Overview

CosyVoice2 is a state-of-the-art multi-lingual text-to-speech provider with **cross-lingual voice synthesis** capabilities, making it ideal for real-time translation scenarios. This provider enables you to speak translated text in the original speaker's voice.

## Features

- ✅ **Cross-lingual Synthesis**: Speak target language with source voice (perfect for translation)
- ✅ **Native Streaming**: <200ms latency with CosyVoice2's bidirectional streaming
- ✅ **Zero-shot Cloning**: Clone any voice from 3-10 seconds of audio
- ✅ **Multi-lingual**: Chinese, English, Japanese, Korean + Chinese dialects
- ✅ **Multiple Modes**: SFT, Zero-shot, Cross-lingual, Instruct
- ✅ **GPU Optimized**: ~4GB VRAM for CosyVoice2-0.5B

## Installation

### 1. Install Dependencies

```bash
# Install sokuji-bridge with TTS support
pip install -e ".[tts]"

# Or install CosyVoice dependencies separately
pip install cosyvoice>=0.0.8 modelscope>=1.11.0 torchaudio>=2.0.0

# Linux GPU support
pip install onnxruntime-gpu==1.18.0

# macOS/Windows
pip install onnxruntime==1.18.0
```

### 2. Download Model

The model will auto-download on first use, or download manually:

```python
from modelscope import snapshot_download

# Recommended: CosyVoice2-0.5B (best streaming performance)
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# Alternative: CosyVoice v1.0 models
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
```

## Configuration

### Basic Translation Configuration

Use the pre-configured translation profile:

```yaml
# configs/cosyvoice_translation.yaml
tts:
  provider: cosyvoice
  device: cuda

  config:
    model: CosyVoice2-0.5B
    inference_mode: cross_lingual  # Optimal for translation

    cross_lingual:
      auto_extract_prompt: true
      prompt_duration_sec: 3.0

    streaming:
      enabled: true

    speed: 1.0
```

### Advanced Configuration

```yaml
tts:
  provider: cosyvoice
  device: cuda

  config:
    # Model Configuration
    model: CosyVoice2-0.5B
    model_dir: pretrained_models/CosyVoice2-0.5B
    auto_download: true
    download_source: modelscope  # or huggingface

    # Default Inference Mode
    inference_mode: cross_lingual  # sft | zero_shot | cross_lingual | instruct

    # Cross-lingual Mode (Translation Scenarios)
    cross_lingual:
      auto_extract_prompt: true
      prompt_duration_sec: 3.0  # Use first N seconds of source audio
      prompt_audio: null  # Provided at runtime

    # Zero-shot Mode (Voice Cloning)
    zero_shot:
      prompt_text: null  # Reference transcript
      prompt_audio: null  # Reference audio
      speaker_embedding_cache: true

    # SFT Mode (Preset Voices)
    sft:
      speaker: default

    # Instruct Mode (Emotion/Style Control)
    instruct:
      instruct_text: "Speak with emotion"
      sft_speaker: default
      prompt_audio: null

    # Streaming Configuration
    streaming:
      enabled: true
      token_hop_len: 50  # Token chunk size
      mel_cache_len: 10  # HiFT cache
      source_cache_len: 10

    # Synthesis Parameters
    speed: 1.0  # 0.5 - 2.0
    text_frontend: true  # Text normalization
```

## Usage

### Translation Workflow (Cross-lingual Mode)

This is the **primary use case** for translation scenarios:

```python
import asyncio
import numpy as np
from providers.tts.cosyvoice_provider import CosyVoiceProvider

async def translation_example():
    # Configuration
    config = {
        "model": "CosyVoice2-0.5B",
        "inference_mode": "cross_lingual",
        "device": "cuda",
        "streaming": {"enabled": True},
    }

    provider = CosyVoiceProvider(config)
    await provider.initialize()

    # Simulate STT output: source audio (e.g., Chinese speech)
    source_audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds at 16kHz

    # Translate: Chinese text → English text (via translation service)
    target_text = "Hello, how are you today?"  # Translated text

    # TTS: Speak English text with Chinese voice
    result = await provider.synthesize(
        text=target_text,
        voice_id="cross_lingual",
        prompt_audio=source_audio,  # Original Chinese audio
        stream=True,
    )

    print(f"Generated {result.duration_ms:.0f}ms of audio")
    # result.audio_data contains the English speech in Chinese voice

    await provider.cleanup()

asyncio.run(translation_example())
```

### Voice Cloning (Zero-shot Mode)

Clone any voice from a short sample:

```python
async def voice_cloning_example():
    config = {
        "model": "CosyVoice2-0.5B",
        "inference_mode": "zero_shot",
        "device": "cuda",
    }

    provider = CosyVoiceProvider(config)
    await provider.initialize()

    # Reference audio and transcript
    reference_audio = np.random.randn(16000 * 5).astype(np.float32)
    reference_text = "This is the reference transcript"

    # Synthesize new text in cloned voice
    result = await provider.synthesize(
        text="New text to synthesize",
        voice_id="zero_shot",
        prompt_audio=reference_audio,
        prompt_text=reference_text,
    )

    await provider.cleanup()

asyncio.run(voice_cloning_example())
```

### Preset Voices (SFT Mode)

Use built-in voices:

```python
async def preset_voice_example():
    config = {
        "model": "CosyVoice-300M-SFT",
        "inference_mode": "sft",
        "device": "cuda",
    }

    provider = CosyVoiceProvider(config)
    await provider.initialize()

    # List available voices
    voices = await provider.get_voices()
    print(f"Available voices: {[v['id'] for v in voices if v['mode'] == 'sft']}")

    # Synthesize with preset voice
    result = await provider.synthesize(
        text="Hello world",
        voice_id="speaker1",  # Or any available speaker
    )

    await provider.cleanup()

asyncio.run(preset_voice_example())
```

### Streaming Synthesis

Process text stream with low latency:

```python
async def streaming_example():
    config = {
        "model": "CosyVoice2-0.5B",
        "inference_mode": "cross_lingual",
        "device": "cuda",
        "streaming": {"enabled": True},
    }

    provider = CosyVoiceProvider(config)
    await provider.initialize()

    # Text stream generator
    async def text_stream():
        chunks = ["First sentence. ", "Second sentence. ", "Third sentence."]
        for chunk in chunks:
            yield chunk

    reference_audio = np.random.randn(16000 * 3).astype(np.float32)

    # Stream synthesis
    async for result in provider.synthesize_stream(
        text_stream(),
        voice_id="cross_lingual",
        prompt_audio=reference_audio,
    ):
        print(f"Got audio chunk: {result.duration_ms:.0f}ms")
        # Play or stream result.audio_data

    await provider.cleanup()

asyncio.run(streaming_example())
```

## Inference Modes

### 1. Cross-lingual (Translation)

**Use Case**: Real-time translation with voice preservation

**Parameters**:
- `text`: Target language text
- `prompt_audio`: Source language audio (16kHz)
- `stream`: Enable streaming (recommended)

**Example**:
```python
result = await provider.synthesize(
    text="Hello",  # English text
    voice_id="cross_lingual",
    prompt_audio=chinese_audio,  # Chinese speaker
    stream=True,
)
```

### 2. Zero-shot (Voice Cloning)

**Use Case**: Clone any voice from short sample

**Parameters**:
- `text`: Text to synthesize
- `prompt_audio`: Reference audio (16kHz)
- `prompt_text`: Reference transcript

**Example**:
```python
result = await provider.synthesize(
    text="New content",
    voice_id="zero_shot",
    prompt_audio=reference_audio,
    prompt_text="Reference transcript",
)
```

### 3. SFT (Preset Voices)

**Use Case**: Fast synthesis with built-in voices

**Parameters**:
- `text`: Text to synthesize
- `speaker`: Speaker ID (from `get_voices()`)

**Example**:
```python
result = await provider.synthesize(
    text="Hello world",
    voice_id="speaker1",
    inference_mode="sft",
)
```

### 4. Instruct (Emotion Control)

**Use Case**: Add emotion/style to synthesis

**Parameters**:
- `text`: Text to synthesize
- `instruct_text`: Style instruction
- `prompt_audio`: Reference audio
- `speaker`: Base speaker

**Example**:
```python
result = await provider.synthesize(
    text="I'm so happy!",
    voice_id="instruct",
    instruct_text="Speak with joy and excitement",
    prompt_audio=reference_audio,
)
```

## Performance

### Latency Benchmarks

| Model | Mode | Latency | VRAM | Quality |
|-------|------|---------|------|---------|
| CosyVoice2-0.5B | Streaming | <200ms | ~4GB | Excellent |
| CosyVoice2-0.5B | Non-streaming | ~500ms | ~4GB | Excellent |
| CosyVoice-300M | Streaming | ~250ms | ~5GB | Very Good |
| CosyVoice-300M | Non-streaming | ~600ms | ~5GB | Very Good |

### Accuracy Improvements (vs CosyVoice 1.0)

- ✅ 30-50% fewer pronunciation errors
- ✅ Better speaker similarity in cross-lingual mode
- ✅ More natural prosody and intonation

## Translation Pipeline Integration

### Full Workflow

```
┌─────────────────────────────────────────────────┐
│ STT (faster-whisper)                            │
│   Input: Chinese speech audio                   │
│   Output: "你好" (transcript) + audio            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ Translation (NLLB)                              │
│   Input: "你好"                                  │
│   Output: "Hello"                               │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ TTS (CosyVoice2 Cross-lingual)                 │
│   Input:                                        │
│     - text: "Hello"                             │
│     - prompt_audio: Chinese speech              │
│   Output: "Hello" in Chinese voice              │
└─────────────────────────────────────────────────┘
```

### Integration Code

```python
async def full_translation_pipeline(source_audio: np.ndarray):
    """Complete translation pipeline with voice preservation"""

    # 1. STT: Transcribe source audio
    stt_result = await stt_provider.transcribe(source_audio)
    source_text = stt_result.text
    source_lang = stt_result.language

    # 2. Translation: Translate to target language
    translation_result = await translation_provider.translate(
        text=source_text,
        source_lang=source_lang,
        target_lang="en",
    )
    target_text = translation_result.translated_text

    # 3. TTS: Synthesize with source voice
    tts_result = await tts_provider.synthesize(
        text=target_text,
        voice_id="cross_lingual",
        prompt_audio=source_audio,  # Reuse source audio!
        stream=True,
    )

    return tts_result
```

## Troubleshooting

### Model Download Issues

```python
# Manual download with mirror
from modelscope import snapshot_download
snapshot_download(
    'iic/CosyVoice2-0.5B',
    local_dir='pretrained_models/CosyVoice2-0.5B',
    revision='master'
)
```

### CUDA Out of Memory

```yaml
# Use smaller model or reduce batch size
config:
  model: CosyVoice-300M  # Instead of larger models
  streaming:
    token_hop_len: 30  # Smaller chunks
```

### Audio Quality Issues

```yaml
# Increase prompt audio duration
cross_lingual:
  prompt_duration_sec: 5.0  # Use more reference audio

# Adjust synthesis speed
speed: 0.95  # Slightly slower for better quality
```

## API Reference

### CosyVoiceProvider

```python
class CosyVoiceProvider(TTSProvider):
    """CosyVoice2 TTS provider with cross-lingual synthesis"""

    async def initialize() -> None:
        """Load model and prepare for synthesis"""

    async def synthesize(
        text: str,
        voice_id: str,
        language: Optional[str] = None,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            language: Optional language hint
            **kwargs:
                - inference_mode: sft|zero_shot|cross_lingual|instruct
                - prompt_audio: Reference audio (numpy array or path)
                - prompt_text: Reference transcript (for zero_shot)
                - instruct_text: Style instruction (for instruct)
                - speed: Speech speed (0.5 - 2.0)
                - stream: Enable streaming (bool)

        Returns:
            SynthesisResult with audio data
        """

    async def synthesize_stream(
        text_stream: AsyncIterator[str],
        voice_id: str,
        **kwargs
    ) -> AsyncIterator[SynthesisResult]:
        """Stream synthesis from text stream"""

    async def get_voices() -> list[Dict[str, Any]]:
        """Get available voices and modes"""

    async def cleanup() -> None:
        """Release resources"""
```

## Best Practices

### For Translation

1. **Use Cross-lingual Mode**: Optimal for maintaining speaker identity
2. **Extract Clean Prompt**: Use 3-10 seconds of clear speech
3. **Enable Streaming**: Reduces latency significantly
4. **Cache Embeddings**: Reuse speaker embeddings when possible

### For Quality

1. **Clean Reference Audio**: Remove background noise
2. **Appropriate Speed**: Keep speed between 0.8-1.2 for naturalness
3. **Text Normalization**: Enable `text_frontend` for better pronunciation
4. **Sufficient Context**: Use at least 3 seconds of prompt audio

### For Performance

1. **GPU Acceleration**: Use CUDA for best performance
2. **Batch Processing**: Process multiple texts together when possible
3. **Model Selection**: CosyVoice2-0.5B offers best speed/quality tradeoff
4. **Streaming**: Always use streaming for real-time applications

## Resources

- **GitHub**: https://github.com/FunAudioLLM/CosyVoice
- **CosyVoice2 Paper**: https://arxiv.org/abs/2412.10117
- **Model Hub**: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- **Demo**: https://funaudiollm.github.io/cosyvoice2/
- **Issues**: https://github.com/kizuna-ai-lab/sokuji-bridge/issues

## License

CosyVoice2 is released under Apache 2.0 license.
