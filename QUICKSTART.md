# Quick Start Guide

Get Sokuji-Bridge running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with 12GB+ VRAM (for local models)
- CUDA 12.1+ or 13.0 (if using GPU)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sokuji-bridge.git
cd sokuji-bridge
```

### 2. Install dependencies

#### Option A: CPU-only (slower, no GPU needed)

```bash
pip install -r requirements.txt
```

#### Option B: GPU-accelerated (recommended)

```bash
# Install PyTorch with CUDA 12.1/13.0 support first
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

**Note**: PyTorch cu121 build is compatible with both CUDA 12.1 and CUDA 13.0.

### 3. Download models

```bash
# The models will be downloaded automatically on first use
# Or manually download:

# Faster-Whisper models (auto-downloaded)
# NLLB models (auto-downloaded from HuggingFace)
# Piper voices - download from: https://github.com/rhasspy/piper/releases
```

## Quick Test

### Test Individual Providers

```bash
python examples/basic_usage.py
```

This will:
1. Test STT provider (faster-whisper)
2. Test Translation provider (NLLB)
3. Test TTS provider (Piper)
4. Run a full end-to-end pipeline test

### Expected Output

```
🌉 Sokuji-Bridge Example

Running individual provider tests...

1️⃣  Testing STT Provider...
  ✓ STT Result: [en] ... (conf: 0.95)

2️⃣  Testing Translation Provider...
  ✓ Translation: Hello, how are you?
    → 你好，你好吗？

3️⃣  Testing TTS Provider...
  ✓ TTS Result: 44100 bytes, 1000ms

All providers tested successfully! ✅

Running full pipeline test...

📊 Performance Metrics:
  STT Latency: 450.3ms
  Translation Latency: 320.5ms
  TTS Latency: 180.2ms
  Total Latency: 1500.0ms

Example completed successfully! 🎉
```

## Configuration Profiles

Sokuji-Bridge comes with 4 pre-configured profiles:

### Fast (Default) - Pure Local

```yaml
STT: faster-whisper (medium)
Translation: NLLB-200 (1.3B)
TTS: Piper (CPU)
Latency: ~1.5-2s
Cost: $0/month
```

### Hybrid - Mixed Local/API

```yaml
STT: faster-whisper (local)
Translation: DeepL API
TTS: Piper (local)
Latency: ~2-3s
Cost: $10-30/month
```

### Quality - Best Quality

```yaml
STT: faster-whisper (large-v3)
Translation: GPT-4o-mini
TTS: XTTS v2
Latency: ~3-5s
Cost: $20-50/month
```

### CPU - No GPU Required

```yaml
STT: faster-whisper (base, int8)
Translation: Google Translate API
TTS: Piper (CPU)
Latency: ~5-10s
Cost: Low
```

## Usage Examples

### Python API

```python
import asyncio
from config.manager import ConfigManager
from core.pipeline import TranslationPipeline
from providers.base import AudioChunk

async def translate_audio():
    # Load configuration
    config = ConfigManager.from_profile("fast").get_config()

    # Create pipeline
    pipeline = TranslationPipeline(...)
    await pipeline.initialize()

    # Process audio
    audio_chunk = AudioChunk(...)
    result = await pipeline.process_single(audio_chunk)

    # Get translated audio
    print(f"Audio: {len(result.audio_data)} bytes")

    await pipeline.cleanup()

asyncio.run(translate_audio())
```

### Configuration File

```python
# Load from custom config file
from pathlib import Path
config = ConfigManager.from_file(Path("my_config.yaml")).get_config()
```

### Switch Providers at Runtime

```python
config_manager = ConfigManager.from_profile("fast")
config_manager.update_provider("translation", "deepl_api", {
    "formality": "more"
})
```

## Common Issues

### 1. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Use smaller models: `model_size: "small"` instead of `"medium"`
- Use int8 quantization: `compute_type: "int8"`
- Switch to CPU: `device: "cpu"`

### 2. Models Not Found

**Problem**: `FileNotFoundError: Model not found`

**Solutions**:
- Models will auto-download on first use
- For Piper: Download voices from https://github.com/rhasspy/piper/releases
- Set `model_cache_dir` in config

### 3. Slow Performance

**Problem**: Translation is very slow

**Solutions**:
- Enable GPU: `device: "cuda"`
- Use float16: `precision: "float16"`
- Reduce beam size: `num_beams: 2`
- Enable batching and caching

### 4. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'faster_whisper'`

**Solution**:
```bash
pip install faster-whisper transformers torch piper-tts
```

## Next Steps

- 📖 Read the [full documentation](./docs/)
- 🐳 Try [Docker deployment](./docker/)
- 🔧 Customize [configuration](./configs/)
- 🧪 Write your own [providers](./docs/custom_providers.md)
- 🚀 Deploy to [production](./docs/deployment.md)

## Performance Tips

1. **GPU Utilization**
   - Use CUDA for STT and Translation
   - Keep TTS on CPU (Piper is CPU-optimized)

2. **Latency Optimization**
   - Enable VAD filtering: `vad_filter: true`
   - Reduce beam size: `beam_size: 2`
   - Use smaller models for development

3. **Memory Optimization**
   - Use int8 quantization for models
   - Limit cache size: `cache_size: 5000`
   - Clear cache periodically

4. **Quality vs Speed**
   - Fast: `model_size: "small"`, `num_beams: 2`
   - Balanced: `model_size: "medium"`, `num_beams: 4`
   - Quality: `model_size: "large"`, `num_beams: 5`

## Support

- 📧 Issues: [GitHub Issues](https://github.com/yourusername/sokuji-bridge/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/sokuji-bridge/discussions)
- 📚 Docs: [Documentation](./docs/)

---

Happy translating! 🌉✨
