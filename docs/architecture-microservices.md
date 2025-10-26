# Microservices Architecture

This document describes the microservices architecture of Sokuji-Bridge for dependency isolation and independent scaling.

## Architecture Overview

Sokuji-Bridge uses a microservices architecture with gRPC communication to solve dependency conflicts between different providers (STT, Translation, TTS).

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/WebSocket
       ▼
┌─────────────────┐
│    Gateway      │  (FastAPI + gRPC clients)
│   Port: 8000    │
└────────┬────────┘
         │ gRPC
    ┌────┴─────┬──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│  STT   │ │ Trans  │ │  TTS   │
│ :50051 │ │ :50052 │ │ :50053 │
└────────┘ └────────┘ └────────┘
```

## Services

### 1. STT Service (Port 50051)
**Purpose**: Speech-to-Text transcription
**Providers**: faster-whisper, whisper-api, azure-stt
**Dependencies**: faster-whisper, silero-vad, torch
**GPU**: Required for local models

**gRPC API**:
- `Transcribe(AudioChunk) → TranscriptionResult`
- `TranscribeStream(stream AudioChunk) → stream TranscriptionResult`
- `GetSupportedLanguages() → LanguageList`
- `HealthCheck() → HealthResponse`
- `GetMetrics() → MetricsResponse`

**Environment Variables**:
- `STT_PROVIDER`: Provider name (default: faster_whisper)
- `MODEL_SIZE`: Model size (tiny, base, small, medium, large)
- `DEVICE`: Device (cuda, cpu, auto)
- `COMPUTE_TYPE`: Compute type (int8, float16, float32)
- `VAD_FILTER`: Enable VAD filtering (true/false)

### 2. Translation Service (Port 50052)
**Purpose**: Text translation
**Providers**: nllb_local, deepl_api, gpt-4, google_translate
**Dependencies**: transformers, torch, sentencepiece
**GPU**: Required for local models

**gRPC API**:
- `Translate(text, src, tgt) → TranslationResult`
- `TranslateBatch([texts], src, tgt) → [TranslationResult]`
- `TranslateStream(stream text) → stream TranslationResult`
- `GetSupportedLanguages() → LanguageList`
- `HealthCheck() → HealthResponse`
- `GetMetrics() → MetricsResponse`

**Environment Variables**:
- `TRANSLATION_PROVIDER`: Provider name (default: nllb_local)
- `MODEL`: Model name (e.g., facebook/nllb-200-distilled-1.3B)
- `DEVICE`: Device (cuda, cpu)
- `PRECISION`: Precision (float16, float32)

### 3. TTS Service (Port 50053)
**Purpose**: Text-to-Speech synthesis
**Providers**: piper, kokoro, xtts, elevenlabs
**Dependencies**: piper-tts, TTS (Coqui)
**GPU**: Optional (required for XTTS)

**gRPC API**:
- `Synthesize(text, voice_id) → SynthesisResult`
- `SynthesizeStream(stream text) → stream SynthesisResult`
- `GetVoices() → VoiceList`
- `GetSupportedLanguages() → LanguageList`
- `HealthCheck() → HealthResponse`
- `GetMetrics() → MetricsResponse`

**Environment Variables**:
- `TTS_PROVIDER`: Provider name (default: piper)
- `MODEL`: Model/voice name (e.g., en_US-lessac-medium)
- `DEVICE`: Device (cpu, cuda)

### 4. Gateway Service (Port 8000)
**Purpose**: API gateway and pipeline orchestration
**Dependencies**: FastAPI, grpcio (no ML libraries)
**GPU**: Not required

**REST API**:
- `GET /` - Service info
- `GET /health` - Health check all services
- `POST /translate/text` - Text-only translation (for testing)
- `WS /ws/translate` - WebSocket real-time translation
- `GET /services/stt/languages` - Get STT supported languages
- `GET /services/translation/languages` - Get translation languages
- `GET /services/tts/voices` - Get available TTS voices

**Environment Variables**:
- `STT_SERVICE_URL`: STT service address (default: localhost:50051)
- `TRANSLATION_SERVICE_URL`: Translation service address (default: localhost:50052)
- `TTS_SERVICE_URL`: TTS service address (default: localhost:50053)
- `PORT`: HTTP port (default: 8000)

## Deployment

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with 12GB+ VRAM (for local models)
- NVIDIA Container Toolkit

### Quick Start

1. **Generate gRPC code** (first time only):
```bash
./scripts/generate_protos.sh
```

2. **Start all services**:
```bash
docker compose up -d
```

3. **Check health**:
```bash
curl http://localhost:8000/health
```

4. **View logs**:
```bash
docker compose logs -f gateway
docker compose logs -f stt-service
docker compose logs -f translation-service
docker compose logs -f tts-service
```

### Development Mode

For development with hot-reload:

```bash
docker compose -f docker compose.yml -f docker compose.dev.yml up
```

This mounts source directories as volumes for live code updates.

### Provider Configuration

Change providers via environment variables in `docker compose.yml`:

**Example: Switch to API-based providers**:
```yaml
services:
  stt-service:
    environment:
      - STT_PROVIDER=whisper_api
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  translation-service:
    environment:
      - TRANSLATION_PROVIDER=deepl_api
      - DEEPL_API_KEY=${DEEPL_API_KEY}

  tts-service:
    environment:
      - TTS_PROVIDER=elevenlabs
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
```

Restart services to apply changes:
```bash
docker compose restart stt-service translation-service tts-service
```

## Benefits

### Dependency Isolation
Each service has its own `requirements.txt` with only necessary dependencies:
- **STT Service**: faster-whisper, silero-vad, torch
- **Translation Service**: transformers, torch, sentencepiece
- **TTS Service**: piper-tts, TTS
- **Gateway**: FastAPI, grpcio (no ML libs)

No more dependency conflicts between providers!

### Independent Scaling
Scale services independently based on resource needs:
```bash
docker compose up -d --scale translation-service=3
```

### Resource Allocation
Allocate GPU resources per service:
```yaml
services:
  stt-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # 1 GPU for STT
              capabilities: [gpu]
```

### Easy Debugging
Restart individual services without affecting others:
```bash
docker compose restart stt-service
```

View service-specific logs:
```bash
docker compose logs -f stt-service
```

### Provider Flexibility
Swap providers without touching code:
```bash
# Switch from faster-whisper to Whisper API
docker compose stop stt-service
# Edit docker compose.yml: STT_PROVIDER=whisper_api
docker compose up -d stt-service
```

## Performance

### Latency
- **gRPC overhead**: ~10-50ms (acceptable)
- **Localhost network**: <5ms between containers
- **Total pipeline**: Still <2s end-to-end

### Throughput
Each service can handle concurrent requests independently.

### Resource Usage
- **STT Service**: 5-8GB VRAM (medium model)
- **Translation Service**: 3-5GB VRAM (1.3B model)
- **TTS Service**: <1GB RAM (Piper on CPU)
- **Gateway**: <500MB RAM

## Monitoring

### Health Checks
Each service exposes health check endpoint:
```bash
# Via gRPC
grpcurl -plaintext localhost:50051 sokuji.stt.STTService/HealthCheck

# Via Gateway HTTP
curl http://localhost:8000/health
```

### Metrics
Each service exposes metrics:
```bash
grpcurl -plaintext localhost:50051 sokuji.stt.STTService/GetMetrics
```

### Logs
Structured logging with loguru:
```bash
docker compose logs -f --tail=100 stt-service
```

## Troubleshooting

### Service Won't Start
Check logs:
```bash
docker compose logs stt-service
```

Common issues:
- GPU not available: Check NVIDIA runtime
- Model download: First start takes time
- Port conflict: Change PORT environment variable

### gRPC Connection Errors
Check service health:
```bash
docker compose ps
curl http://localhost:8000/health
```

Restart services:
```bash
docker compose restart
```

### Dependency Conflicts
Each service has isolated dependencies. If issues persist:
```bash
docker compose build --no-cache stt-service
```

## Migration from Monolithic

The monolithic pipeline is preserved in `src/core/pipeline.py` for reference.

**Key Changes**:
1. Providers now wrapped by gRPC services
2. Pipeline orchestration moved to Gateway
3. Dependencies separated per service
4. Docker-based deployment instead of single Python process

## Future Enhancements

- [ ] Add authentication/authorization
- [ ] Implement load balancing for high availability
- [ ] Add service mesh (Istio) for advanced routing
- [ ] Kubernetes deployment manifests
- [ ] Distributed tracing (Jaeger/Zipkin)
- [ ] Advanced monitoring (Grafana/Prometheus)
