# Translation Service Requirements

This directory contains Python dependencies for different Translation providers.

## Provider Mapping

Each requirements file corresponds to a provider implementation in `src/providers/translation/`:

| Requirements File | Provider Implementation | Description |
|-------------------|------------------------|-------------|
| `base.txt` | N/A (shared) | Common dependencies for all Translation providers (grpcio, loguru, pydantic) |
| `nllb_local.txt` | `src/providers/translation/nllb_provider.py` | Meta NLLB-200 local model (200+ languages, GPU-accelerated) |
| `deepl_api.txt` | `src/providers/translation/deepl_provider.py` | DeepL API (cloud-based, high quality) |
| `openai_gpt4.txt` | `src/providers/translation/openai_gpt4_provider.py` | OpenAI GPT-4 API (context-aware translation) |
| `google_translate.txt` | `src/providers/translation/google_translate_provider.py` | Google Translate API (cloud-based) |

## Docker Multi-Stage Build

The `Dockerfile` uses multi-stage builds to isolate provider dependencies:

```dockerfile
# Base stage: Install common dependencies
FROM python:3.11-slim AS base
COPY services/translation_service/requirements/base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/base.txt

# Provider stages: One per provider
FROM base AS provider-nllb_local
COPY services/translation_service/requirements/nllb_local.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/nllb_local.txt

# Final stage: Select provider via ARG
ARG PROVIDER=nllb_local
FROM provider-${PROVIDER} AS final
```

## Adding New Translation Providers

To add a new Translation provider:

1. **Create provider implementation** in `src/providers/translation/`:
   ```python
   # src/providers/translation/new_provider.py
   from src.providers.base import TranslationProvider

   class NewTranslationProvider(TranslationProvider):
       async def translate(self, text, source_lang, target_lang, context=None):
           # Implementation
           pass
   ```

2. **Create requirements file** in this directory:
   ```bash
   # services/translation_service/requirements/new_provider.txt
   some-translation-library==1.0.0
   dependency-one>=2.0
   ```

3. **Add Dockerfile stage** in `services/translation_service/Dockerfile`:
   ```dockerfile
   FROM base AS provider-new_provider
   COPY services/translation_service/requirements/new_provider.txt /tmp/
   RUN pip install --no-cache-dir -r /tmp/new_provider.txt
   ```

4. **Update server.py** in `services/translation_service/server.py`:
   ```python
   def load_provider() -> TranslationProvider:
       provider_name = os.getenv("TRANSLATION_PROVIDER", "nllb_local")

       if provider_name == "new_provider":
           from src.providers.translation.new_provider import NewTranslationProvider
           return NewTranslationProvider(config)
   ```

5. **Update documentation**:
   - Add provider to `.env.example`
   - Add provider to `docs/PROVIDERS.md`
   - Update this README

## Switching Providers

Providers are selected at **build time** via environment variables:

```bash
# In .env file
TRANSLATION_PROVIDER=nllb_local

# Rebuild service
docker compose build translation-service

# Restart service
docker compose up -d translation-service
```

## Provider Comparison

| Provider | Type | Languages | Quality | Speed | Cost |
|----------|------|-----------|---------|-------|------|
| NLLB Local | Local | 200+ | Good | Fast | GPU required |
| DeepL API | Cloud | 30+ | Excellent | Fast | Pay per char |
| OpenAI GPT-4 | Cloud | 100+ | Excellent | Moderate | Pay per token |
| Google Translate | Cloud | 130+ | Good | Fast | Pay per char |

See `docs/PROVIDERS.md` for detailed comparison and selection guide.
