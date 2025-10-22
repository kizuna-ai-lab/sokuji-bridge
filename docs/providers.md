# Provider Reference Guide

This document lists all supported providers for STT, Translation, and TTS services, along with their requirements and configuration.

## How Provider Selection Works

Sokuji-Bridge uses **build-time provider selection** through Docker multi-stage builds. When you build a service, you specify which provider to use, and only that provider's dependencies are included in the final image.

### Quick Start

1. **Copy environment file**:
```bash
cp .env.example .env
```

2. **Edit `.env` to select providers**:
```bash
STT_PROVIDER=faster_whisper
TRANSLATION_PROVIDER=nllb_local
TTS_PROVIDER=piper
```

3. **Build and start**:
```bash
./scripts/build_services.sh
docker compose up -d
```

### Switching Providers

To switch a provider after initial deployment:

```bash
# 1. Update .env file
echo "TTS_PROVIDER=xtts" >> .env

# 2. Rebuild the service
docker compose build tts-service

# 3. Restart the service
docker compose up -d tts-service
```

---

## STT (Speech-to-Text) Providers

### `faster_whisper` (Default) ⭐

**Type**: Local Model, GPU-accelerated
**Best for**: High accuracy, privacy, offline operation

**Pros**:
- ✅ Excellent accuracy
- ✅ No API costs
- ✅ Privacy (fully local)
- ✅ Multi-language support (100+ languages)
- ✅ Built-in VAD (Voice Activity Detection)

**Cons**:
- ❌ Requires GPU (12GB+ VRAM)
- ❌ Slower than API options
- ❌ Large model files (~1-3GB)

**Configuration**:
```bash
STT_PROVIDER=faster_whisper
STT_MODEL_SIZE=medium        # tiny, base, small, medium, large, large-v3
STT_DEVICE=cuda              # cuda or cpu
STT_COMPUTE_TYPE=float16     # int8, float16, float32
STT_VAD_FILTER=true
```

**Resource Requirements**:
- GPU: 5-8GB VRAM (medium model)
- RAM: 2GB
- Disk: 1-3GB (model files)

---

### `whisper_api`

**Type**: Cloud API (OpenAI)
**Best for**: Low latency, no GPU required

**Pros**:
- ✅ Fast response time
- ✅ No local GPU needed
- ✅ Always up-to-date models
- ✅ Pay-as-you-go pricing

**Cons**:
- ❌ API costs (~$0.006/minute)
- ❌ Requires internet connection
- ❌ Data sent to OpenAI

**Configuration**:
```bash
STT_PROVIDER=whisper_api
OPENAI_API_KEY=your_api_key_here
```

**Resource Requirements**:
- GPU: None
- RAM: <500MB
- Disk: <100MB

**Cost**: ~$0.006 per minute of audio

---

### `azure_stt`

**Type**: Cloud API (Microsoft Azure)
**Best for**: Enterprise deployments, Azure integration

**Pros**:
- ✅ Enterprise SLA
- ✅ Real-time streaming support
- ✅ Custom speech models
- ✅ Compliance certifications

**Cons**:
- ❌ API costs
- ❌ Requires Azure account
- ❌ More complex setup

**Configuration**:
```bash
STT_PROVIDER=azure_stt
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=your_azure_region  # e.g., eastus
```

**Resource Requirements**:
- GPU: None
- RAM: <500MB
- Disk: <100MB

**Cost**: ~$1 per audio hour

---

## Translation Providers

### `nllb_local` (Default) ⭐

**Type**: Local Model (Meta's NLLB), GPU-accelerated
**Best for**: Multilingual, privacy, no API costs

**Pros**:
- ✅ Free (no API costs)
- ✅ 200+ languages
- ✅ Privacy (fully local)
- ✅ Good quality for most languages

**Cons**:
- ❌ Requires GPU
- ❌ Lower quality than DeepL/GPT-4 for some pairs
- ❌ Large model files (~1-3GB)

**Configuration**:
```bash
TRANSLATION_PROVIDER=nllb_local
TRANSLATION_MODEL=facebook/nllb-200-distilled-1.3B  # or larger models
TRANSLATION_DEVICE=cuda
TRANSLATION_PRECISION=float16
```

**Resource Requirements**:
- GPU: 3-5GB VRAM
- RAM: 2GB
- Disk: 1-3GB

---

### `deepl_api`

**Type**: Cloud API (DeepL)
**Best for**: Highest quality translation

**Pros**:
- ✅ Excellent translation quality
- ✅ Fast response
- ✅ Natural-sounding output
- ✅ No GPU required

**Cons**:
- ❌ API costs
- ❌ Limited language pairs (30+)
- ❌ Requires internet

**Configuration**:
```bash
TRANSLATION_PROVIDER=deepl_api
DEEPL_API_KEY=your_deepl_api_key
```

**Resource Requirements**:
- GPU: None
- RAM: <500MB
- Disk: <100MB

**Cost**: Free tier: 500,000 characters/month, then ~$5-25/million characters

---

### `openai_gpt4`

**Type**: Cloud API (OpenAI GPT-4)
**Best for**: Best quality, context-aware translation

**Pros**:
- ✅ Highest quality translation
- ✅ Context-aware
- ✅ Can handle idioms and nuances
- ✅ Many language pairs

**Cons**:
- ❌ Higher API costs
- ❌ Slower than dedicated translation APIs
- ❌ Requires OpenAI API key

**Configuration**:
```bash
TRANSLATION_PROVIDER=openai_gpt4
OPENAI_API_KEY=your_openai_api_key
```

**Resource Requirements**:
- GPU: None
- RAM: <500MB
- Disk: <100MB

**Cost**: ~$0.01-0.06 per request (depends on text length)

---

### `google_translate`

**Type**: Cloud API (Google)
**Best for**: Wide language support, free tier

**Pros**:
- ✅ 130+ languages
- ✅ Free tier available
- ✅ Fast response
- ✅ Reliable

**Cons**:
- ❌ Lower quality than DeepL/GPT-4
- ❌ Rate limits on free tier
- ❌ Requires internet

**Configuration**:
```bash
TRANSLATION_PROVIDER=google_translate
# No API key needed for basic usage
```

**Resource Requirements**:
- GPU: None
- RAM: <500MB
- Disk: <100MB

**Cost**: Free tier: 500,000 characters/month, then ~$20/million characters

---

## TTS (Text-to-Speech) Providers

### `piper` (Default) ⭐

**Type**: Local Model, CPU-optimized
**Best for**: Fast, free, low resource usage

**Pros**:
- ✅ Very fast (real-time on CPU)
- ✅ Free (no API costs)
- ✅ No GPU required
- ✅ Many voices available
- ✅ Low resource usage

**Cons**:
- ❌ Lower quality than XTTS/ElevenLabs
- ❌ Less natural-sounding
- ❌ Limited voice customization

**Configuration**:
```bash
TTS_PROVIDER=piper
TTS_MODEL=en_US-lessac-medium  # or other voice models
TTS_DEVICE=cpu
```

**Resource Requirements**:
- GPU: None
- RAM: <1GB
- Disk: <500MB

---

### `xtts`

**Type**: Local Model (Coqui XTTS v2), GPU-accelerated
**Best for**: High quality, voice cloning

**Pros**:
- ✅ Excellent quality
- ✅ Voice cloning support
- ✅ Multilingual
- ✅ Free (no API costs)

**Cons**:
- ❌ Requires GPU
- ❌ Slower than Piper
- ❌ Large model files (~2GB)

**Configuration**:
```bash
TTS_PROVIDER=xtts
TTS_DEVICE=cuda
```

**Resource Requirements**:
- GPU: 4-6GB VRAM
- RAM: 2GB
- Disk: ~2GB

---

### `elevenlabs`

**Type**: Cloud API (ElevenLabs)
**Best for**: Highest quality, most natural voices

**Pros**:
- ✅ Best-in-class quality
- ✅ Very natural-sounding
- ✅ Voice cloning
- ✅ Emotion control

**Cons**:
- ❌ API costs
- ❌ Requires internet
- ❌ Character limits on free tier

**Configuration**:
```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

**Resource Requirements**:
- GPU: None
- RAM: <500MB
- Disk: <100MB

**Cost**: Free tier: 10,000 characters/month, then ~$5-330/month

---

### `kokoro`

**Type**: Local Model (Experimental)
**Best for**: Experimental/research

**Pros**:
- ✅ Open source
- ✅ Free

**Cons**:
- ❌ Experimental status
- ❌ Limited documentation
- ❌ May require manual installation

**Configuration**:
```bash
TTS_PROVIDER=kokoro
```

---

## Deployment Scenarios

### Scenario 1: Full Local (No API Costs)

**Best for**: Privacy, offline, high GPU availability

```bash
STT_PROVIDER=faster_whisper
TRANSLATION_PROVIDER=nllb_local
TTS_PROVIDER=xtts
```

**Requirements**:
- GPU: 12GB+ VRAM
- Cost: $0/month
- Latency: ~2-3s

---

### Scenario 2: Full Cloud (No GPU)

**Best for**: No GPU, willing to pay for quality

```bash
STT_PROVIDER=whisper_api
TRANSLATION_PROVIDER=deepl_api
TTS_PROVIDER=elevenlabs
```

**Requirements**:
- GPU: None
- Cost: ~$20-100/month (usage-based)
- Latency: ~1-2s

---

### Scenario 3: Hybrid (Balanced)

**Best for**: Optimize cost vs performance

```bash
STT_PROVIDER=faster_whisper   # Local (slowest step, worth GPU)
TRANSLATION_PROVIDER=deepl_api # Cloud (fastest, best quality)
TTS_PROVIDER=piper              # Local CPU (fast enough)
```

**Requirements**:
- GPU: 8GB VRAM
- Cost: ~$10-30/month (DeepL only)
- Latency: ~1.5-2s

---

## Provider Comparison Matrix

| Provider | Type | GPU | Cost/Month | Quality | Latency |
|----------|------|-----|------------|---------|---------|
| **STT** |
| faster_whisper | Local | Yes | $0 | ⭐⭐⭐⭐⭐ | 500ms |
| whisper_api | Cloud | No | ~$10-50 | ⭐⭐⭐⭐⭐ | 200ms |
| azure_stt | Cloud | No | ~$30-100 | ⭐⭐⭐⭐ | 150ms |
| **Translation** |
| nllb_local | Local | Yes | $0 | ⭐⭐⭐⭐ | 300ms |
| deepl_api | Cloud | No | ~$5-25 | ⭐⭐⭐⭐⭐ | 100ms |
| openai_gpt4 | Cloud | No | ~$20-100 | ⭐⭐⭐⭐⭐ | 500ms |
| google_translate | Cloud | No | ~$5-20 | ⭐⭐⭐ | 150ms |
| **TTS** |
| piper | Local | No | $0 | ⭐⭐⭐ | 100ms |
| xtts | Local | Yes | $0 | ⭐⭐⭐⭐ | 300ms |
| elevenlabs | Cloud | No | ~$5-50 | ⭐⭐⭐⭐⭐ | 200ms |
| kokoro | Local | Maybe | $0 | ⭐⭐ | Unknown |

---

## Adding New Providers

To add a new provider:

1. **Create requirements file**:
```bash
touch services/stt_service/requirements/my_provider.txt
```

2. **Add dependencies**:
```txt
my-provider-lib>=1.0.0
```

3. **Add Dockerfile stage**:
```dockerfile
FROM base as provider-my_provider
COPY services/stt_service/requirements/my_provider.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/my_provider.txt
```

4. **Implement provider class** in `src/providers/stt/my_provider.py`

5. **Update server.py** to load the new provider

6. **Test**:
```bash
STT_PROVIDER=my_provider docker compose build stt-service
```

---

## Troubleshooting

### Provider Build Fails

```bash
# Check which provider is being built
docker compose config | grep -A 5 "args:"

# Try building with --no-cache
docker compose build --no-cache stt-service
```

### Provider Not Found Error

Make sure the provider name in `.env` matches exactly:
```bash
# Correct
STT_PROVIDER=faster_whisper

# Incorrect (will fail)
STT_PROVIDER=faster-whisper  # Use underscore, not hyphen
STT_PROVIDER=FasterWhisper    # Must be lowercase with underscore
```

### GPU Not Available

For GPU providers (faster_whisper, nllb_local, xtts):
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

---

## Next Steps

- [Quick Start Guide](../QUICKSTART_MICROSERVICES.md)
- [Architecture Documentation](MICROSERVICES.md)
- [Configuration Guide](CONFIGURATION.md)
