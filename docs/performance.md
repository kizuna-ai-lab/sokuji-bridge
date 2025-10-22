# Performance Tuning Guide

Optimize Sokuji-Bridge for your specific latency, accuracy, and resource requirements.

## Performance Overview

Default configuration achieves:
- **End-to-end latency**: 1.5-2 seconds
- **STT accuracy**: 90%+ (medium model)
- **Translation quality**: High (NLLB 1.3B)
- **TTS quality**: Natural (Piper medium)
- **Resource usage**: ~6GB VRAM, ~8GB RAM

## Optimization Strategies

### 1. Latency Optimization

#### Target: <1 Second End-to-End

**STT Optimization:**
```yaml
# docker-compose.yml
services:
  stt-service:
    environment:
      - MODEL_SIZE=tiny       # Fastest model (200ms)
      - COMPUTE_TYPE=int8     # Faster inference
      - BEAM_SIZE=1          # Disable beam search
      - VAD_FILTER=true      # Process only speech
```

**Translation Optimization:**
```yaml
services:
  translation-service:
    environment:
      - TRANSLATION_MODEL=facebook/nllb-200-distilled-600M  # Smaller model
      - MAX_LENGTH=50                                        # Limit output length
      - NUM_BEAMS=1                                          # Greedy decoding
```

**TTS Optimization:**
```yaml
services:
  tts-service:
    environment:
      - TTS_MODEL=en_US-lessac-low     # Fastest model
      - SAMPLE_RATE=16000               # Lower sample rate
```

**Result**: ~800ms latency, slightly reduced quality

#### Target: <500ms (Ultra-Low Latency)

Use cloud APIs for parallel processing:
```yaml
services:
  stt-service:
    environment:
      - STT_PROVIDER=azure
      - AZURE_SPEECH_KEY=${AZURE_KEY}

  translation-service:
    environment:
      - TRANSLATION_PROVIDER=deepl
      - DEEPL_API_KEY=${DEEPL_KEY}

  tts-service:
    environment:
      - TTS_PROVIDER=elevenlabs
      - ELEVENLABS_API_KEY=${ELEVENLABS_KEY}
```

**Result**: ~400ms latency, $0.02/minute cost

### 2. Accuracy Optimization

#### Maximum Quality Configuration

**STT High Accuracy:**
```yaml
services:
  stt-service:
    environment:
      - MODEL_SIZE=large-v3        # Best accuracy
      - COMPUTE_TYPE=float16       # Full precision
      - BEAM_SIZE=5               # Beam search
      - BEST_OF=5                 # Multiple attempts
      - TEMPERATURE=0.0           # Deterministic
      - VAD_FILTER=true           # Clean audio
      - VAD_THRESHOLD=0.5         # Conservative VAD
```

**Translation Excellence:**
```yaml
services:
  translation-service:
    environment:
      - TRANSLATION_MODEL=facebook/nllb-200-3.3B  # Largest model
      - PRECISION=float32                          # Maximum precision
      - NUM_BEAMS=5                                # Beam search
      - LENGTH_PENALTY=1.0                         # Balanced length
```

**TTS Premium:**
```yaml
services:
  tts-service:
    environment:
      - TTS_MODEL=en_US-libritts_r-medium  # Studio quality
      - SAMPLE_RATE=48000                  # High fidelity
      - AUDIO_FORMAT=wav                   # Lossless
```

**Result**: 3-4s latency, 95%+ accuracy, ~12GB VRAM

### 3. Resource Optimization

#### CPU-Only Mode

```yaml
services:
  stt-service:
    environment:
      - DEVICE=cpu
      - COMPUTE_TYPE=int8
      - MODEL_SIZE=base
      - NUM_WORKERS=4
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G

  translation-service:
    environment:
      - DEVICE=cpu
      - PRECISION=int8
      - TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

  tts-service:
    environment:
      - DEVICE=cpu
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
```

**Result**: 4-5s latency, no GPU required

#### Memory-Constrained (4GB VRAM)

```yaml
services:
  stt-service:
    environment:
      - MODEL_SIZE=small
      - COMPUTE_TYPE=int8
      - BATCH_SIZE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 2G

  translation-service:
    environment:
      - TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
      - PRECISION=int8
      - MAX_BATCH_SIZE=1
    deploy:
      resources:
        limits:
          memory: 1.5G

  tts-service:
    environment:
      - TTS_MODEL=en_US-lessac-low
    deploy:
      resources:
        limits:
          memory: 500M
```

### 4. Scaling Optimization

#### Horizontal Scaling

```yaml
services:
  stt-service:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure

  gateway:
    environment:
      - LOAD_BALANCE_STRATEGY=round_robin
      - HEALTH_CHECK_INTERVAL=5
      - CIRCUIT_BREAKER_THRESHOLD=5
```

#### Service Mesh Configuration

```yaml
# docker-compose.prod.yml
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - gateway

  gateway:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### 5. Network Optimization

#### gRPC Tuning

```yaml
services:
  gateway:
    environment:
      - GRPC_MAX_MESSAGE_LENGTH=10485760  # 10MB
      - GRPC_KEEPALIVE_TIME=10            # 10s
      - GRPC_KEEPALIVE_TIMEOUT=5          # 5s
      - GRPC_KEEPALIVE_PERMIT=true
      - GRPC_HTTP2_MAX_PINGS=2
```

#### WebSocket Configuration

```yaml
services:
  gateway:
    environment:
      - WS_MAX_MESSAGE_SIZE=1048576  # 1MB
      - WS_PING_INTERVAL=30           # 30s
      - WS_PING_TIMEOUT=10            # 10s
      - WS_MAX_CONNECTIONS=100
      - WS_COMPRESSION=true
```

## Model Selection Guide

### STT Models

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| tiny | 39x | 75% | 1GB | Real-time demos |
| base | 16x | 85% | 1GB | Low-latency apps |
| small | 6x | 89% | 2GB | Balanced |
| medium | 2x | 92% | 5GB | **Default** |
| large | 1x | 94% | 10GB | Transcription |
| large-v3 | 1x | 95% | 10GB | Maximum accuracy |

### Translation Models

| Model | Speed | Quality | Memory | Use Case |
|-------|-------|---------|--------|----------|
| NLLB-600M | Fast | Good | 1.2GB | Low-latency |
| NLLB-1.3B | Medium | Better | 2.6GB | **Default** |
| NLLB-3.3B | Slow | Best | 6.6GB | Quality focus |
| DeepL API | Fast | Excellent | 0 | Cloud-based |
| GPT-4 | Slow | Perfect | 0 | Context-aware |

### TTS Models

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| piper-low | Fast | Basic | 30MB | Testing |
| piper-medium | Medium | Good | 60MB | **Default** |
| piper-high | Slow | Better | 120MB | Quality |
| kokoro | Slow | Natural | 2GB | Japanese |
| elevenlabs | Medium | Premium | 0 | Cloud-based |

## Monitoring & Metrics

### Key Performance Indicators

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gateway'
    static_configs:
      - targets: ['gateway:8000']
    metrics_path: '/metrics'

  - job_name: 'stt'
    static_configs:
      - targets: ['stt-service:50051']

  - job_name: 'translation'
    static_configs:
      - targets: ['translation-service:50052']

  - job_name: 'tts'
    static_configs:
      - targets: ['tts-service:50053']
```

### Metrics to Monitor

1. **Latency Metrics**:
   - End-to-end latency (p50, p95, p99)
   - Per-service latency breakdown
   - Queue wait times

2. **Throughput Metrics**:
   - Requests per second
   - Audio minutes processed
   - Characters translated

3. **Resource Metrics**:
   - GPU utilization
   - VRAM usage
   - CPU and RAM usage
   - Network bandwidth

4. **Quality Metrics**:
   - STT confidence scores
   - Translation confidence
   - Audio quality (SNR)

### Performance Dashboard

```bash
# Start monitoring stack
docker compose -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

## Optimization Workflow

### 1. Baseline Measurement
```bash
# Run performance test
python tests/performance/benchmark.py --duration 60

# Results saved to: benchmarks/baseline.json
```

### 2. Profile Services
```bash
# Profile STT service
docker exec stt-service python -m cProfile -o profile.pstats server.py

# Analyze profile
python -m pstats profile.pstats
```

### 3. Apply Optimizations
- Start with model size adjustments
- Then optimize compute types
- Finally tune batch sizes and concurrency

### 4. Validate Changes
```bash
# Run A/B test
python tests/performance/ab_test.py \
  --config-a configs/baseline.yml \
  --config-b configs/optimized.yml
```

## Common Performance Issues

### High Latency

**Symptoms**: >3s end-to-end latency

**Solutions**:
1. Reduce model sizes
2. Enable VAD filtering
3. Decrease beam size
4. Use GPU acceleration
5. Optimize network settings

### Memory Leaks

**Symptoms**: Increasing memory usage over time

**Solutions**:
1. Enable memory limits in Docker
2. Implement request pooling
3. Clear model cache periodically
4. Monitor with memory profiler

### GPU Underutilization

**Symptoms**: <50% GPU usage

**Solutions**:
1. Increase batch size
2. Enable dynamic batching
3. Use multiple worker processes
4. Optimize data loading pipeline

## Production Checklist

- [ ] Set appropriate model sizes for latency requirements
- [ ] Configure resource limits and reservations
- [ ] Enable health checks and circuit breakers
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling policies
- [ ] Implement request rate limiting
- [ ] Enable gRPC keepalive settings
- [ ] Configure WebSocket compression
- [ ] Set up log aggregation
- [ ] Test failover scenarios

## Advanced Optimizations

### Custom Model Quantization
```python
# Quantize Whisper model to int8
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
```

### ONNX Runtime Acceleration
```yaml
services:
  translation-service:
    environment:
      - USE_ONNX=true
      - ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

### TensorRT Optimization
```bash
# Convert model to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

## See Also

- [Architecture Documentation](./architecture.md)
- [Provider Configuration](./providers.md)
- [Monitoring Guide](./MICROSERVICES.md#monitoring)
- [Debugging Guide](./DEBUG_MODE.md)