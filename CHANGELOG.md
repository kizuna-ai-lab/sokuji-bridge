# Changelog

All notable changes to sokuji-bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### CosyVoice2 TTS Provider (v0.1.0)

- **Cross-lingual Voice Synthesis**: New TTS provider optimized for translation scenarios
  - Speak translated text in original speaker's voice
  - Perfect for real-time voice translation applications
  - Uses CosyVoice2-0.5B model with native streaming support

- **Multiple Inference Modes**:
  - `cross_lingual`: Translate while preserving voice (optimal for translation)
  - `zero_shot`: Clone any voice from 3-10 seconds of audio
  - `sft`: Fast synthesis with preset voices
  - `instruct`: Emotion and style-controlled synthesis

- **Performance Features**:
  - Native streaming with <200ms latency
  - 30-50% fewer pronunciation errors vs CosyVoice 1.0
  - GPU-optimized (~4GB VRAM for CosyVoice2-0.5B)
  - Supports CPU inference for local deployment

- **Multi-lingual Support**:
  - Chinese (Mandarin + dialects: Cantonese, Sichuanese, etc.)
  - English, Japanese, Korean
  - Cross-lingual synthesis between any supported language pairs

- **Configuration**:
  - New `configs/cosyvoice_translation.yaml` for translation scenarios
  - Updated `configs/default.yaml` with CosyVoice examples
  - Auto-download support via ModelScope or HuggingFace

- **Documentation**:
  - Comprehensive guide: `docs/cosyvoice_provider.md`
  - Example scripts: `examples/cosyvoice_example.py`
  - Complete unit test coverage (>80%)

- **Dependencies**:
  - Added CosyVoice support to `pyproject.toml`
  - `cosyvoice>=0.0.8`
  - `modelscope>=1.11.0` for model downloads
  - `torchaudio>=2.0.0`
  - `onnxruntime-gpu` (Linux) / `onnxruntime` (macOS/Windows)

#### Integration

- Seamless integration with existing STT and Translation providers
- Auto-extract source audio as prompt for voice preservation
- Compatible with pipeline streaming architecture

### Technical Details

**Translation Workflow**:
```
STT (source audio + text)
  → Translation (target text)
  → TTS (target text + source audio as prompt)
  → Output (target text in source voice)
```

**Performance Benchmarks**:
- Cross-lingual latency: <200ms (streaming)
- Model memory: ~4GB VRAM (CosyVoice2-0.5B)
- Supported models:
  - CosyVoice2-0.5B (recommended)
  - CosyVoice-300M, CosyVoice-300M-SFT, CosyVoice-300M-Instruct

### Files Added

- `src/providers/tts/cosyvoice_provider.py` - Main provider implementation
- `src/providers/tts/__init__.py` - Provider registration
- `configs/cosyvoice_translation.yaml` - Translation-optimized config
- `docs/cosyvoice_provider.md` - Complete usage guide
- `examples/cosyvoice_example.py` - Example scripts
- `tests/providers/tts/test_cosyvoice_provider.py` - Unit tests

### Files Modified

- `pyproject.toml` - Added CosyVoice dependencies
- `configs/default.yaml` - Added CosyVoice configuration examples

### Migration Notes

To use CosyVoice2 TTS:

1. Install dependencies:
   ```bash
   pip install -e ".[tts]"
   ```

2. Download model (auto or manual):
   ```python
   from modelscope import snapshot_download
   snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
   ```

3. Update config:
   ```yaml
   tts:
     provider: cosyvoice
     device: cuda
     config:
       model: CosyVoice2-0.5B
       inference_mode: cross_lingual
   ```

See `docs/cosyvoice_provider.md` for detailed setup and usage.

---

## [0.1.0] - 2025-01-XX

### Initial Release

- Basic pipeline architecture (STT → Translation → TTS)
- Piper TTS provider
- faster-whisper STT provider
- NLLB translation provider
- YAML-based configuration system
- Docker support
- Basic monitoring and metrics

[Unreleased]: https://github.com/kizuna-ai-lab/sokuji-bridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kizuna-ai-lab/sokuji-bridge/releases/tag/v0.1.0
