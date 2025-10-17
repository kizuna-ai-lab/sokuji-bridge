# Sokuji-Bridge

**Real-time Voice Translation System** with modular STT, Translation, and TTS pipelines.

[中文文档](./docs/README_zh.md)

## 🌟 Features

- **Modular Architecture**: Swap any STT, Translation, or TTS provider easily
- **Low Latency**: <2 seconds end-to-end latency with local providers
- **Docker-First**: Microservice architecture for flexible deployment
- **Multiple Configurations**: Fast (local), Hybrid (mixed), Quality (API) profiles
- **Streaming Pipeline**: Asynchronous processing for minimal delay
- **Smart Segmentation**: VAD-based intelligent audio chunking

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with 12GB+ VRAM (for local providers)
- CUDA 11.8+ and cuDNN

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
sokuji translate --source zh --target en --profile fast

# Translate audio file
sokuji translate --source zh --target en --input audio.mp3 --output translated.wav

# Start as API server
sokuji serve --host 0.0.0.0 --port 8000
```

## 📊 Provider Configurations

### Profile A: Fast Local (Default)
```yaml
STT: faster-whisper (medium)
Translation: NLLB-200 (1.3B)
TTS: Piper (CPU)
Latency: 1.5-2s | Cost: $0/month | VRAM: ~5GB
```

### Profile B: Hybrid Quality
```yaml
STT: faster-whisper (local)
Translation: DeepL API (cloud)
TTS: Piper (local)
Latency: 2-3s | Cost: $10-30/month
```

### Profile C: Maximum Quality
```yaml
STT: faster-whisper (large-v3)
Translation: GPT-4o-mini API
TTS: XTTS v2 (voice cloning)
Latency: 3-5s | Cost: $20-50/month | VRAM: ~10GB
```

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
- [Piper](https://github.com/rhasspy/piper) for fast TTS
- All the amazing open-source AI model creators

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/sokuji-bridge/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/sokuji-bridge/discussions)

---

Made with ❤️ for the real-time translation community
