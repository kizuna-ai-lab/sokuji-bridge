# Sokuji-Bridge

**Real-time Voice Translation System** with modular STT, Translation, and TTS pipelines.

[中文文档](./docs/README_zh.md)

## 🌟 Features

- **Modular Architecture**: Swap any STT, Translation, or TTS provider easily
- **Low Latency**: <2 seconds end-to-end latency with local providers
- **Docker-First**: Microservice architecture for flexible deployment
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Streaming Pipeline**: Asynchronous processing for minimal delay
- **Smart Segmentation**: VAD-based intelligent audio chunking

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with 12GB+ VRAM (for local providers)
- CUDA 12.1+ or 13.0 (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sokuji-bridge.git
cd sokuji-bridge

# Option 1: Docker (Recommended)
docker-compose up -d

# Option 2: Local installation
pip install -e ".[all]"
python scripts/download_models.sh
```

### Usage

```bash
# Start translation (microphone input)
sokuji translate --source zh --target en

# Translate audio file
sokuji translate --source zh --target en --input audio.mp3 --output translated.wav

# Start as API server
sokuji serve --host 0.0.0.0 --port 8000
```

## ⚙️ Configuration

Sokuji-Bridge uses a YAML-based configuration system. The default configuration (`configs/default.yaml`) is optimized for low latency with local models:

```yaml
stt:
  provider: faster_whisper
  config:
    model_size: medium
    device: cuda
    vad_filter: true  # Built-in VAD filtering

translation:
  provider: nllb_local
  config:
    model: facebook/nllb-200-distilled-1.3B
    device: cuda

tts:
  provider: piper
  config:
    model: en_US-lessac-medium
```

**Performance:** 1.5-2s latency | $0/month | ~5GB VRAM

### Customization

Edit `configs/default.yaml` to customize providers, models, or parameters:

```bash
# Use larger STT model for better accuracy
stt:
  config:
    model_size: large-v3  # medium → large-v3

# Use cloud translation for better quality
translation:
  provider: deepl_api  # nllb_local → deepl_api
```

See [Configuration Guide](./docs/CONFIGURATION.md) for all options.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Audio Input Stream                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            VAD + Intelligent Segmentation                │
│            (Silero VAD / WebRTC VAD)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  STT Service (gRPC)                      │
│         faster-whisper | Whisper API | Azure             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Translation Service (gRPC)                  │
│            NLLB | DeepL API | GPT-4 | Google             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   TTS Service (gRPC)                     │
│          Piper | Kokoro | XTTS | ElevenLabs              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Audio Output Stream                     │
└─────────────────────────────────────────────────────────┘
```

## 📖 Documentation

- [Installation Guide](./docs/installation.md)
- [Configuration](./docs/configuration.md)
- [Provider Guide](./docs/providers.md)
- [API Reference](./docs/api.md)
- [Performance Tuning](./docs/performance.md)
- [Docker Deployment](./docs/docker.md)

## 🔧 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src tests
ruff check src tests

# Type checking
mypy src
```

## 📝 Project Structure

```
sokuji-bridge/
├── src/
│   ├── core/              # Pipeline orchestrator
│   ├── services/          # gRPC microservices
│   ├── providers/         # STT/Translation/TTS providers
│   ├── utils/             # VAD, audio processing, monitoring
│   └── api/               # FastAPI + WebSocket
├── configs/               # Configuration files
├── proto/                 # gRPC protocol definitions
├── docker/                # Docker images
├── scripts/               # Deployment and utility scripts
└── tests/                 # Unit, integration, E2E tests
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for fast STT
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) for multilingual translation
- [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) for fast local TTS
- All the amazing open-source AI model creators

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/sokuji-bridge/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/sokuji-bridge/discussions)

---

Made with ❤️ for the real-time translation community
