# Installation Guide

Complete installation guide for Sokuji-Bridge.

## Quick Install (Recommended)

### Step 1: Install PyTorch with CUDA Support

```bash
# For CUDA 12.1 / 13.0 (your system)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Sokuji-Bridge as Package

```bash
# Install in editable mode with all dependencies
pip install -e ".[all]"
```

This will:
- ✅ Install all required dependencies
- ✅ Set up correct import paths
- ✅ Allow you to edit code and see changes immediately
- ✅ Enable the `sokuji` command

### Step 3: Verify Installation

```bash
# Test configuration system
python examples/simple_test.py
```

---

## Alternative: Manual Installation

If you want to install dependencies manually:

```bash
# 1. Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Install package in development mode
pip install -e .
```

---

## Common Issues

### Issue 1: ImportError with relative imports

**Error**:
```
ImportError: attempted relative import beyond top-level package
```

**Cause**: Running scripts directly without installing the package.

**Solution**: Install the package first!
```bash
pip install -e .
```

Then import like this:
```python
from sokuji_bridge.config.manager import ConfigManager
from sokuji_bridge.core.pipeline import TranslationPipeline
```

Or run examples as modules:
```bash
python -m examples.basic_usage
```

### Issue 2: CUDA not available

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution 1**: Check CUDA is detected
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

**Solution 2**: Use smaller models
Edit `configs/default.yaml`:
```yaml
stt:
  config:
    model_size: "small"  # Instead of "medium"
    compute_type: "int8"  # Instead of "float16"
```

**Solution 3**: Use CPU-only
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Models not found

**Error**:
```
FileNotFoundError: Model not found
```

**Solution**: Models download automatically on first use. Make sure you have:
- ✅ Internet connection
- ✅ ~5-10GB free disk space
- ✅ Patience (first download takes 5-10 minutes)

Models are cached in:
- **Whisper/NLLB**: `~/.cache/huggingface/`
- **Piper TTS**: `models/piper/` (project directory) or `~/.local/share/piper/`

**Manual Piper Model Download**:

If automatic download fails, you can manually download Piper models:

```bash
# Download a specific voice model
python -m piper.download_voices en_US-lessac-medium --download-dir models/piper

# Download other voices
python -m piper.download_voices zh_CN-huayan-medium --download-dir models/piper
python -m piper.download_voices ja_JP-natsu-medium --download-dir models/piper

# List all available voices
python -m piper.download_voices
```

Available voices: [Piper Voices on Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main)

---

## Directory Structure After Installation

```
sokuji-bridge/
├── src/                          # Source code
│   └── sokuji_bridge/           # Importable as package
├── examples/
│   ├── simple_test.py           # Quick test (no model download)
│   └── basic_usage.py           # Full example
├── configs/
│   └── default.yaml             # Configuration
└── sokuji_bridge.egg-info/      # Package metadata (auto-generated)
```

---

## Usage After Installation

### Python API

```python
from sokuji_bridge.config.manager import ConfigManager
from sokuji_bridge.core.pipeline import TranslationPipeline
from sokuji_bridge.providers.stt.faster_whisper_provider import FasterWhisperProvider
from sokuji_bridge.providers.translation.nllb_provider import NLLBProvider
from sokuji_bridge.providers.tts.piper_provider import PiperProvider

# Load configuration
config = ConfigManager.from_profile("fast").get_config()

# Create providers
stt = FasterWhisperProvider(config.stt.config)
translator = NLLBProvider(config.translation.config)
tts = PiperProvider(config.tts.config)

# Initialize
await stt.initialize()
await translator.initialize()
await tts.initialize()

# Use...
```

### Run Examples

```bash
# Simple test (no model download)
python examples/simple_test.py

# Full example (downloads models)
python examples/basic_usage.py
```

---

## Uninstall

```bash
pip uninstall sokuji-bridge
```

Models remain cached in `~/.cache/sokuji-bridge/` for reuse.

---

## Development Setup

For contributors:

```bash
# 1. Clone repository
git clone https://github.com/yourusername/sokuji-bridge.git
cd sokuji-bridge

# 2. Install with development dependencies
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Run tests
pytest

# 5. Format code
black src tests
ruff check src tests --fix
```

---

## Next Steps

After successful installation:

1. 📖 Read [QUICKSTART.md](./QUICKSTART.md) for usage examples
2. 🔧 Customize [configs/default.yaml](./configs/default.yaml)
3. 🧪 Run tests: `python examples/simple_test.py`
4. 🚀 Build your translation app!

---

**Need help?** Open an issue on GitHub!
