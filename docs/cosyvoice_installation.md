# CosyVoice Installation Guide

## ⚠️ Important Note

The CosyVoice package on PyPI (`cosyvoice==0.0.8`) is a **simplified API wrapper** and does not include the full CosyVoice2 model code. For full functionality, you need to install from the official GitHub repository.

## Installation Methods

### Method 1: Install from GitHub (Recommended)

```bash
# Clone the official repository
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Update submodules if needed
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Install CosyVoice in development mode
pip install -e .
```

### Method 2: Direct pip install from GitHub

```bash
pip install git+https://github.com/FunAudioLLM/CosyVoice.git
```

### Method 3: Using PyPI package (Limited功能)

```bash
# This installs a simplified wrapper - not recommended for full features
pip install cosyvoice>=0.0.8
```

## Verify Installation

```python
# Test if CosyVoice2 is properly installed
from cosyvoice.cli.cosyvoice import CosyVoice2
print("✅ CosyVoice2 installed successfully!")
```

If you see `ModuleNotFoundError: No module named 'cosyvoice.cli'`, you are using the PyPI wrapper. Please install from GitHub instead.

## Download Models

### Automatic Download (Recommended)

The provider will automatically download models on first use:

```python
from providers.tts.cosyvoice_provider import CosyVoiceProvider

config = {
    "model": "CosyVoice2-0.5B",
    "auto_download": True,  # Enable auto-download
    "download_source": "modelscope",  # or "huggingface"
}

provider = CosyVoiceProvider(config)
await provider.initialize()  # Model will be downloaded automatically
```

### Manual Download

#### Option 1: ModelScope (Recommended for China)

```python
from modelscope import snapshot_download

# CosyVoice2-0.5B (recommended)
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# Alternative models
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
```

#### Option 2: HuggingFace

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FunAudioLLM/CosyVoice2-0.5B",
    local_dir="pretrained_models/CosyVoice2-0.5B"
)
```

#### Option 3: Git LFS

```bash
# Install git-lfs if not already installed
git lfs install

# Clone model repository
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# Or from HuggingFace
git clone https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B pretrained_models/CosyVoice2-0.5B
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 - 3.11
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for CosyVoice2-0.5B)
- **CUDA**: 11.8 or 12.1
- **RAM**: 8GB+
- **Storage**: ~5GB for model files

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 12.1
- **RAM**: 16GB+

## Dependencies

### Core Dependencies (automatically installed)

```
torch>=2.0.0
torchaudio>=2.0.0
onnxruntime-gpu==1.18.0  # Linux
onnxruntime==1.18.0      # macOS/Windows
modelscope>=1.11.0
transformers
numpy
scipy
```

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get install -y sox libsox-dev ffmpeg
```

#### CentOS/RHEL
```bash
sudo yum install -y sox sox-devel ffmpeg
```

#### macOS
```bash
brew install sox ffmpeg
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cosyvoice.cli'`

**Problem**: You are using the PyPI wrapper package, not the full repository.

**Solution**: Install from GitHub:
```bash
pip uninstall cosyvoice
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -e .
```

### Issue: CUDA Out of Memory

**Solution 1**: Use smaller model
```yaml
config:
  model: CosyVoice-300M  # Instead of CosyVoice2-0.5B
```

**Solution 2**: Reduce batch size
```yaml
streaming:
  token_hop_len: 30  # Smaller chunks
```

### Issue: Model download fails

**Solution**: Manual download with alternative source
```bash
# Try HuggingFace if ModelScope fails
export HF_ENDPOINT=https://hf-mirror.com  # Mirror for faster download
huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir pretrained_models/CosyVoice2-0.5B
```

### Issue: Import errors after installation

**Solution**: Ensure all submodules are initialized
```bash
cd CosyVoice
git submodule update --init --recursive
pip install -e . --force-reinstall
```

## Verification Script

Save this as `test_cosyvoice.py`:

```python
#!/usr/bin/env python3
"""Test CosyVoice installation"""

import sys

print("Testing CosyVoice installation...")

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print("✅ CosyVoice2 can be imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Please install from GitHub: git clone https://github.com/FunAudioLLM/CosyVoice.git")
    sys.exit(1)

# Test 2: Check dependencies
print("\n2. Testing dependencies...")
try:
    import torch
    import torchaudio
    import onnxruntime
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ torchaudio: {torchaudio.__version__}")
    print(f"✅ ONNX Runtime: {onnxruntime.__version__}")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# Test 3: CUDA check
print("\n3. Testing CUDA...")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA not available, will use CPU (slower)")

# Test 4: Model check
print("\n4. Testing model files...")
from pathlib import Path
model_path = Path("pretrained_models/CosyVoice2-0.5B")
if model_path.exists():
    required_files = ["cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt"]
    missing = [f for f in required_files if not (model_path / f).exists()]
    if missing:
        print(f"❌ Missing model files: {missing}")
    else:
        print("✅ All model files present")
else:
    print(f"⚠️  Model not found at {model_path}")
    print("   Run: python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')\"")

print("\n✅ Installation test complete!")
```

Run it:
```bash
python test_cosyvoice.py
```

## Next Steps

After successful installation:

1. **Run examples**: `python examples/cosyvoice_example.py`
2. **Read documentation**: `docs/cosyvoice_provider.md`
3. **Configure for translation**: Use `configs/cosyvoice_translation.yaml`

## Support

- **GitHub Issues**: https://github.com/FunAudioLLM/CosyVoice/issues
- **Documentation**: https://github.com/FunAudioLLM/CosyVoice#readme
- **sokuji-bridge Issues**: https://github.com/kizuna-ai-lab/sokuji-bridge/issues
