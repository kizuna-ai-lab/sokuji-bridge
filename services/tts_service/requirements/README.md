# TTS Service Requirements

This directory contains Python dependencies for different TTS (Text-to-Speech) providers.

## Provider Mapping

Each requirements file corresponds to a provider implementation in `src/providers/tts/`:

| Requirements File | Provider Implementation | Description |
|-------------------|------------------------|-------------|
| `base.txt` | N/A (shared) | Common dependencies for all TTS providers (grpcio, loguru, numpy, pydantic, soundfile) |
| `piper.txt` | `src/providers/tts/piper_provider.py` | Piper TTS (CPU-optimized, fast, lightweight) |
| `xtts.txt` | `src/providers/tts/xtts_provider.py` | XTTS v2 (GPU-accelerated, voice cloning, high quality) |
| `elevenlabs.txt` | `src/providers/tts/elevenlabs_provider.py` | ElevenLabs API (cloud-based, premium quality) |
| `kokoro.txt` | `src/providers/tts/kokoro_provider.py` | Kokoro TTS (experimental, local) |

## Docker Multi-Stage Build

The `Dockerfile` uses multi-stage builds to isolate provider dependencies:

```dockerfile
# Base stage: Install common dependencies (CPU-based)
FROM python:3.11-slim AS base
COPY services/tts_service/requirements/base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/base.txt

# Provider stages: One per provider
FROM base AS provider-piper
COPY services/tts_service/requirements/piper.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/piper.txt

# XTTS uses CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS xtts-base
RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg libsndfile1
COPY services/tts_service/requirements/base.txt /tmp/base.txt
COPY services/tts_service/requirements/xtts.txt /tmp/xtts.txt
RUN pip3 install --no-cache-dir -r /tmp/base.txt -r /tmp/xtts.txt

FROM xtts-base AS provider-xtts

# Final stage: Select provider via ARG
ARG PROVIDER=piper
FROM provider-${PROVIDER} AS final
```

## Adding New TTS Providers

To add a new TTS provider:

1. **Create provider implementation** in `src/providers/tts/`:
   ```python
   # src/providers/tts/new_provider.py
   from src.providers.base import TTSProvider

   class NewTTSProvider(TTSProvider):
       async def synthesize(self, text, voice_id, language=None, **kwargs):
           # Implementation
           pass
   ```

2. **Create requirements file** in this directory:
   ```bash
   # services/tts_service/requirements/new_provider.txt
   some-tts-library==1.0.0
   dependency-one>=2.0
   ```

3. **Add Dockerfile stage** in `services/tts_service/Dockerfile`:
   ```dockerfile
   FROM base AS provider-new_provider
   COPY services/tts_service/requirements/new_provider.txt /tmp/
   RUN pip install --no-cache-dir -r /tmp/new_provider.txt
   ```

   **Note**: If your provider needs GPU support (like XTTS), use `nvidia/cuda` base image instead.

4. **Update server.py** in `services/tts_service/server.py`:
   ```python
   def load_provider() -> TTSProvider:
       provider_name = os.getenv("TTS_PROVIDER", "piper")

       if provider_name == "new_provider":
           from src.providers.tts.new_provider import NewTTSProvider
           return NewTTSProvider(config)
   ```

5. **Update documentation**:
   - Add provider to `.env.example`
   - Add provider to `docs/PROVIDERS.md`
   - Update this README

## Switching Providers

Providers are selected at **build time** via environment variables:

```bash
# In .env file
TTS_PROVIDER=piper

# Rebuild service
docker compose build tts-service

# Restart service
docker compose up -d tts-service
```

## Provider Comparison

| Provider | Type | Device | Quality | Speed | Voice Cloning | Cost |
|----------|------|--------|---------|-------|---------------|------|
| Piper | Local | CPU | Good | Very Fast | No | Free |
| XTTS v2 | Local | GPU | Excellent | Moderate | Yes | GPU required |
| ElevenLabs | Cloud | N/A | Premium | Fast | Yes | Pay per char |
| Kokoro | Local | CPU/GPU | Experimental | Fast | No | Free |

## Special Considerations

### GPU-based Providers (XTTS)

XTTS requires:
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Sufficient VRAM (8GB+ recommended)

The Dockerfile uses a special CUDA base image and the `docker-compose.yml` includes GPU resource reservation:

```yaml
tts-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Cloud-based Providers (ElevenLabs)

Cloud providers require API keys set in environment variables:

```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_api_key_here
```

See `docs/PROVIDERS.md` for detailed comparison and selection guide.
