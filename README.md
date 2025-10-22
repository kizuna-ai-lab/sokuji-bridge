# Sokuji-Bridge

**Real-time Voice Translation System** with modular STT, Translation, and TTS pipelines.

## 🌟 Features

- **Modular Architecture**: Swap any STT, Translation, or TTS provider easily
- **Low Latency**: <2 seconds end-to-end latency with local providers
- **Docker-First**: Microservice architecture for flexible deployment
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Streaming Pipeline**: Asynchronous processing for minimal delay
- **Smart Segmentation**: VAD-based intelligent audio chunking

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (v2.0+)
- NVIDIA GPU with 8GB+ VRAM (for local providers)
- NVIDIA Container Toolkit (for GPU acceleration)
- CUDA 12.1+ (installed in containers)

### Installation & Deployment

```bash
# Clone the repository
git clone https://github.com/yourusername/sokuji-bridge.git
cd sokuji-bridge

# Start all microservices with Docker Compose
docker compose up -d

# Check service health
curl http://localhost:8000/health
```

### API Usage

```bash
# Translate text (REST API)
curl -X POST http://localhost:8000/translate/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界",
    "source_language": "zh",
    "target_language": "en",
    "voice_id": "default"
  }'

# WebSocket real-time translation
# Connect to ws://localhost:8000/ws/translate
# See examples/ for client implementations
```

### Service Management

```bash
# View service logs
docker compose logs -f gateway
docker compose logs -f stt-service

# Restart specific service
docker compose restart stt-service

# Stop all services
docker compose down

# Rebuild after code changes
docker compose build stt-service
docker compose up -d stt-service
```

## ⚙️ Configuration

Sokuji-Bridge uses environment variables for service configuration. Edit `.env` or `docker-compose.yml`:

```yaml
# STT Service Configuration
STT_PROVIDER: faster_whisper
MODEL_SIZE: medium          # tiny, base, small, medium, large, large-v3
DEVICE: cuda                # cuda or cpu
COMPUTE_TYPE: float16       # float16, int8, float32
VAD_FILTER: true           # Enable VAD filtering

# Translation Service Configuration
TRANSLATION_PROVIDER: nllb_local
TRANSLATION_MODEL: facebook/nllb-200-distilled-1.3B
PRECISION: float16

# TTS Service Configuration
TTS_PROVIDER: piper
TTS_MODEL: en_US-lessac-medium
```

**Default Performance:** 1.5-2s latency | $0/month | ~6GB VRAM

### Customization

Modify `docker-compose.yml` environment variables:

```bash
# Use larger STT model for better accuracy
services:
  stt-service:
    environment:
      - MODEL_SIZE=large-v3  # Better accuracy, higher latency

# Use CPU-only mode (no GPU required)
services:
  stt-service:
    environment:
      - DEVICE=cpu
      - COMPUTE_TYPE=int8
    deploy:
      resources: {}  # Remove GPU requirement
```

See service-specific READMEs in `services/` for advanced configuration.

## 🏗️ Architecture

### Microservices Design

Sokuji-Bridge uses a **microservices architecture** with gRPC for inter-service communication:

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Applications                        │
│              (REST API / WebSocket / gRPC)                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   Gateway Service (FastAPI)                   │
│         • REST/WebSocket API                                  │
│         • Request routing & orchestration                     │
│         • Health monitoring                                   │
│         Port: 8000                                            │
└─────┬──────────────────┬──────────────────┬──────────────────┘
      │                  │                  │
      │ gRPC             │ gRPC             │ gRPC
      ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ STT Service  │  │ Translation  │  │ TTS Service  │
│              │  │   Service    │  │              │
│ Port: 50051  │  │ Port: 50052  │  │ Port: 50053  │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ Providers:   │  │ Providers:   │  │ Providers:   │
│ • faster-    │  │ • NLLB       │  │ • Piper      │
│   whisper    │  │ • DeepL API  │  │ • Kokoro     │
│ • Whisper    │  │ • GPT-4      │  │ • XTTS       │
│   API        │  │ • Google     │  │ • ElevenLabs │
│ • Azure      │  │              │  │              │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ Features:    │  │ Features:    │  │ Features:    │
│ • VAD filter │  │ • Batch      │  │ • Voice      │
│ • Streaming  │  │ • Streaming  │  │   selection  │
│ • Multi-lang │  │ • Context    │  │ • Streaming  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Key Benefits

- **Scalability**: Scale services independently based on load
- **Resilience**: Service failures don't crash entire system
- **Flexibility**: Swap providers without affecting other services
- **Development**: Teams can develop services independently
- **Deployment**: Deploy updates to individual services

## 🌐 API Endpoints

### Gateway Service (Port 8000)

```bash
# Health check (all services)
GET /health
Response: {"status": "healthy", "services": {...}}

# Translate text end-to-end
POST /translate/text
Body: {
  "text": "Hello world",
  "source_language": "en",
  "target_language": "zh",
  "voice_id": "default"
}

# WebSocket real-time translation
WS /ws/translate
Config: {"source_language": "zh", "target_language": "en", "voice_id": "default"}

# Get supported languages
GET /services/stt/languages
GET /services/translation/languages
GET /services/tts/voices
```

### gRPC Services (Internal)

```protobuf
// STT Service (Port 50051)
service STTService {
  rpc Transcribe(TranscribeRequest) returns (TranscriptionResult);
  rpc TranscribeStream(stream AudioChunk) returns (stream TranscriptionResult);
  rpc HealthCheck(Empty) returns (HealthCheckResponse);
}

// Translation Service (Port 50052)
service TranslationService {
  rpc Translate(TranslateRequest) returns (TranslationResult);
  rpc TranslateBatch(TranslateBatchRequest) returns (TranslateBatchResponse);
  rpc HealthCheck(Empty) returns (HealthCheckResponse);
}

// TTS Service (Port 50053)
service TTSService {
  rpc Synthesize(SynthesizeRequest) returns (SynthesisResult);
  rpc SynthesizeStream(stream SynthesizeRequest) returns (stream SynthesisResult);
  rpc HealthCheck(Empty) returns (HealthCheckResponse);
}
```

## 📖 Documentation

### Getting Started
- [Installation Guide](./docs/installation.md) - Complete installation instructions
- [Quick Start (Microservices)](./docs/quickstart.md) - Get running with Docker in 5 minutes
- [Quick Start (Monolithic)](./docs/quickstart-monolithic.md) - Legacy single-process setup

### Architecture & API
- [Architecture Documentation](./docs/architecture.md) - Microservices architecture details
- [API Reference](./docs/api.md) - Complete REST/WebSocket/gRPC API documentation
- [Provider Guide](./docs/providers.md) - STT, Translation, and TTS provider configuration

### Advanced Topics
- [Performance Tuning](./docs/performance.md) - Optimization strategies and benchmarks
- [Development Guide](./docs/development.md) - Project status and contribution guidelines
- [Audio I/O System](./docs/audio_io.md) - Audio input/output implementation details

### Troubleshooting
- [Microservices Guide](./docs/MICROSERVICES.md) - Detailed microservices documentation
- [VAD Configuration](./docs/VAD_CONFIGURATION.md) - Voice Activity Detection setup
- [Debug Mode](./docs/DEBUG_MODE.md) - Debugging tools and techniques

## 🔧 Development

### Local Development Setup

```bash
# Run services in development mode
docker compose -f docker-compose.dev.yml up

# Run individual service locally
cd services/stt_service
pip install -r requirements/faster_whisper.txt
python server.py

# Run tests
pytest tests/

# Code formatting
black src/ services/ tests/
ruff check src/ services/ tests/

# Type checking
mypy src/
```

### Adding New Providers

1. Create provider class in `src/providers/{stt|translation|tts}/`
2. Implement required abstract methods from base provider
3. Register provider in `src/providers/__init__.py`
4. Add provider requirements to service `requirements/`
5. Update service Dockerfile with new provider target
6. Test with `pytest tests/providers/`

See [Development Guide](./docs/development.md) for details.

## 📝 Project Structure

```
sokuji-bridge/
├── src/
│   ├── generated/         # Generated gRPC code from proto/
│   ├── providers/         # Provider implementations
│   │   ├── stt/          # Speech-to-Text providers
│   │   ├── translation/  # Translation providers
│   │   └── tts/          # Text-to-Speech providers
│   ├── utils/            # Shared utilities
│   └── config/           # Configuration management
├── services/
│   ├── gateway/          # API Gateway (FastAPI + WebSocket)
│   ├── stt_service/      # STT microservice (gRPC server)
│   ├── translation_service/  # Translation microservice
│   └── tts_service/      # TTS microservice
├── proto/                # Protocol Buffer definitions
├── examples/             # Usage examples
├── tests/                # Test suites
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker-compose.yml    # Production deployment
├── docker-compose.dev.yml # Development environment
└── scripts/              # Utility scripts
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
