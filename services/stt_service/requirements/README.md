# STT Service Requirements

This directory contains Python dependencies for different STT (Speech-to-Text) providers.

## Provider Mapping

Each requirements file corresponds to a provider implementation in `src/providers/stt/`:

| Requirements File | Provider Implementation | Description |
|-------------------|------------------------|-------------|
| `base.txt` | N/A (shared) | Common dependencies for all STT providers (grpcio, loguru, numpy, pydantic) |
| `faster_whisper.txt` | `src/providers/stt/faster_whisper_provider.py` | Faster-Whisper local model with GPU support |
| `whisper_api.txt` | `src/providers/stt/whisper_api_provider.py` | OpenAI Whisper API (cloud-based) |
| `azure_stt.txt` | `src/providers/stt/azure_stt_provider.py` | Azure Cognitive Services Speech-to-Text |

## Docker Multi-Stage Build

The `Dockerfile` uses multi-stage builds to isolate provider dependencies:

```dockerfile
# Base stage: Install common dependencies
FROM python:3.11-slim AS base
COPY services/stt_service/requirements/base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/base.txt

# Provider stages: One per provider
FROM base AS provider-faster_whisper
COPY services/stt_service/requirements/faster_whisper.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/faster_whisper.txt

# Final stage: Select provider via ARG
ARG PROVIDER=faster_whisper
FROM provider-${PROVIDER} AS final
```

## Adding New STT Providers

To add a new STT provider:

1. **Create provider implementation** in `src/providers/stt/`:
   ```python
   # src/providers/stt/new_provider.py
   from src.providers.base import STTProvider

   class NewSTTProvider(STTProvider):
       async def transcribe(self, audio, language=None):
           # Implementation
           pass
   ```

2. **Create requirements file** in this directory:
   ```bash
   # services/stt_service/requirements/new_provider.txt
   some-stt-library==1.0.0
   dependency-one>=2.0
   ```

3. **Add Dockerfile stage** in `services/stt_service/Dockerfile`:
   ```dockerfile
   FROM base AS provider-new_provider
   COPY services/stt_service/requirements/new_provider.txt /tmp/
   RUN pip install --no-cache-dir -r /tmp/new_provider.txt
   ```

4. **Update server.py** in `services/stt_service/server.py`:
   ```python
   def load_provider() -> STTProvider:
       provider_name = os.getenv("STT_PROVIDER", "faster_whisper")

       if provider_name == "new_provider":
           from src.providers.stt.new_provider import NewSTTProvider
           return NewSTTProvider(config)
   ```

5. **Update documentation**:
   - Add provider to `.env.example`
   - Add provider to `docs/PROVIDERS.md`
   - Update this README

## Switching Providers

Providers are selected at **build time** via environment variables:

```bash
# In .env file
STT_PROVIDER=faster_whisper

# Rebuild service
docker compose build stt-service

# Restart service
docker compose up -d stt-service
```

See `QUICKSTART_MICROSERVICES.md` for detailed instructions.
