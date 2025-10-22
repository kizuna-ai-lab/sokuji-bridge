# Quick Start Guide - Microservices Architecture

This guide will help you get the microservices architecture up and running in minutes.

## Prerequisites

- Docker & Docker Compose installed
- NVIDIA GPU with 12GB+ VRAM (for local models)
- NVIDIA Container Toolkit installed
- Git (for cloning the repository)

## Installation

### 1. Clone and Navigate

```bash
git clone https://github.com/yourusername/sokuji-bridge.git
cd sokuji-bridge
```

### 2. Configure Providers (Optional)

Copy the environment template and customize providers:

```bash
cp .env.example .env
# Edit .env to select providers (defaults: faster_whisper, nllb_local, piper)
```

See [Provider Reference](./providers.md) for all available providers.

### 3. Generate gRPC Code

Generate Python code from protocol buffer definitions:

```bash
./scripts/generate_protos.sh
```

This creates the gRPC client/server code in `src/generated/`.

### 4. Build Docker Images

**The build script will automatically use providers from your `.env` file**:

```bash
./scripts/build_services.sh
```

This will build services with your selected providers. The script shows which providers are being used.

Or manually with docker compose:

```bash
docker compose build
```

### 4. Start Services

Start all services in the background:

```bash
docker compose up -d
```

Wait for services to initialize (first start downloads models):

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop following logs.

### 5. Verify Health

Check that all services are healthy:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "stt": "ready",
    "translation": "ready",
    "tts": "ready"
  }
}
```

## Usage

### REST API Example

Translate text (without audio):

```bash
curl -X POST http://localhost:8000/translate/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界",
    "source_language": "zh",
    "target_language": "en",
    "voice_id": "default"
  }'
```

Response:
```json
{
  "transcription": "你好世界",
  "transcription_language": "zh",
  "translation": "Hello world",
  "translation_language": "en",
  "audio_duration_ms": 1234.5,
  "total_latency_ms": 567.8
}
```

### WebSocket API Example

For real-time audio translation, use WebSocket:

```python
import asyncio
import websockets
import json

async def translate_audio():
    async with websockets.connect('ws://localhost:8000/ws/translate') as ws:
        # Send configuration
        await ws.send(json.dumps({
            "source_language": "zh",
            "target_language": "en",
            "voice_id": "default",
            "sample_rate": 16000
        }))

        # Send audio chunks
        with open('audio.raw', 'rb') as f:
            chunk = f.read(4096)
            while chunk:
                await ws.send(chunk)

                # Receive translated audio
                audio_response = await ws.recv()
                metadata = await ws.recv()

                print(json.loads(metadata))
                chunk = f.read(4096)

asyncio.run(translate_audio())
```

### Query Available Resources

**Get supported languages (STT)**:
```bash
curl http://localhost:8000/services/stt/languages
```

**Get supported languages (Translation)**:
```bash
curl http://localhost:8000/services/translation/languages
```

**Get available voices (TTS)**:
```bash
curl http://localhost:8000/services/tts/voices
```

## Configuration

### Switching Providers

Sokuji-Bridge supports multiple providers for each service. You can easily switch providers by editing `.env` and rebuilding.

#### Available Providers

**STT (Speech-to-Text)**:
- `faster_whisper` (default) - Local, GPU, high quality
- `whisper_api` - OpenAI API, no GPU needed
- `azure_stt` - Azure API

**Translation**:
- `nllb_local` (default) - Local, GPU, 200+ languages
- `deepl_api` - DeepL API, best quality
- `openai_gpt4` - GPT-4 API
- `google_translate` - Google API

**TTS (Text-to-Speech)**:
- `piper` (default) - Local, CPU, fast
- `xtts` - Local, GPU, high quality
- `elevenlabs` - ElevenLabs API, best quality
- `kokoro` - Experimental

See [Provider Reference](docs/PROVIDERS.md) for detailed comparison.

#### How to Switch

**Example: Switch TTS from Piper to XTTS**

1. **Edit `.env` file**:
```bash
# Change this line:
TTS_PROVIDER=piper
# To:
TTS_PROVIDER=xtts
```

2. **Rebuild the service**:
```bash
docker compose build tts-service
```

3. **Restart the service**:
```bash
docker compose up -d tts-service
```

**Example: Switch to all-cloud deployment (no GPU needed)**:

```bash
# Edit .env
STT_PROVIDER=whisper_api
TRANSLATION_PROVIDER=deepl_api
TTS_PROVIDER=elevenlabs
OPENAI_API_KEY=your_key
DEEPL_API_KEY=your_key
ELEVENLABS_API_KEY=your_key

# Rebuild all services
docker compose build

# Restart
docker compose up -d
```

### Resource Allocation

Adjust GPU allocation in `docker compose.yml`:

```yaml
services:
  stt-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1          # Number of GPUs
              device_ids: ['0']  # Specific GPU IDs
              capabilities: [gpu]
```

## Development

### Development Mode

Start services with volume mounts for live code updates:

```bash
docker compose -f docker compose.yml -f docker compose.dev.yml up
```

Changes to Python files are reflected immediately (no rebuild needed).

### View Logs

**All services**:
```bash
docker compose logs -f
```

**Specific service**:
```bash
docker compose logs -f stt-service
docker compose logs -f translation-service
docker compose logs -f tts-service
docker compose logs -f gateway
```

**Last 100 lines**:
```bash
docker compose logs --tail=100 gateway
```

### Restart Services

**All services**:
```bash
docker compose restart
```

**Specific service**:
```bash
docker compose restart stt-service
```

### Rebuild After Code Changes

If you change requirements or Docker configuration:

```bash
docker compose build stt-service
docker compose up -d stt-service
```

## Troubleshooting

### Service won't start

1. Check logs:
```bash
docker compose logs stt-service
```

2. Common issues:
   - **GPU not available**: Install NVIDIA Container Toolkit
   - **Port already in use**: Change PORT in docker compose.yml
   - **Model download timeout**: Check internet connection, wait longer

### Health check fails

```bash
# Check service status
docker compose ps

# Check gateway health
curl http://localhost:8000/health
```

### Dependency conflicts

Each service has isolated dependencies. If issues persist:

```bash
# Rebuild without cache
docker compose build --no-cache stt-service

# Or rebuild all services
docker compose build --no-cache
```

### Out of memory (GPU)

Reduce model size or batch size:

```yaml
services:
  stt-service:
    environment:
      - MODEL_SIZE=small  # Instead of medium
```

## Stopping Services

**Stop all services**:
```bash
docker compose down
```

**Stop but keep volumes (models)**:
```bash
docker compose stop
```

**Remove everything including volumes**:
```bash
docker compose down -v
```

## Next Steps

- Read [MICROSERVICES.md](docs/MICROSERVICES.md) for architecture details
- Explore [API Reference](docs/api.md) for complete endpoint documentation
- See [Configuration Guide](docs/configuration.md) for advanced configuration
- Check [Performance Tuning](docs/performance.md) for optimization tips

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/sokuji-bridge/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/sokuji-bridge/discussions)

---

**Pro Tips**:
- Use `docker compose logs -f --tail=100` to quickly check recent logs
- Set `LOG_LEVEL=DEBUG` in dev mode for detailed logs
- Monitor GPU usage: `watch -n 1 nvidia-smi`
- Check service memory: `docker stats`
