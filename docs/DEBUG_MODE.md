# Debug Mode for Real-time Translation

This document explains how to use the debug mode in `examples/microphone_to_speaker.py` to validate and analyze translation quality.

## Overview

Debug mode provides comprehensive logging and recording capabilities:

- **Audio Recording**: Saves output audio chunks as WAV files
- **Text Logging**: Records transcriptions and translations in JSONL format
- **Performance Metrics**: Tracks latency for each pipeline stage
- **Event Timeline**: Detailed event log with timestamps

## Basic Usage

### Enable Debug Mode

```bash
# Basic debug mode (saves logs and metrics only)
python examples/microphone_to_speaker.py --debug

# Save audio output as well
python examples/microphone_to_speaker.py --debug --save-audio

# Custom debug directory
python examples/microphone_to_speaker.py --debug --debug-dir ./my_debug
```

### Complete Example

```bash
# Run 60 seconds of Chinese→English translation with full debug
python examples/microphone_to_speaker.py \
    --source zh \
    --target en \
    --duration 60 \
    --debug \
    --save-audio
```

## Debug Output Structure

Debug mode creates the following directory structure:

```
debug_output/
├── audio/                  # Audio recordings (if --save-audio)
│   ├── output_20250120_143022_0001.wav
│   ├── output_20250120_143022_0002.wav
│   └── ...
├── text/                   # Text logs
│   └── events_20250120_143022.jsonl
└── metrics/                # Performance metrics
    └── metrics_20250120_143022.json
```

### Audio Files

Output audio chunks saved as 16-bit PCM WAV files (16kHz, mono):

- **Format**: `output_<session_id>_<chunk_number>.wav`
- **Purpose**: Verify TTS quality and translation accuracy

### Events File (JSONL)

Each line is a JSON event with timestamp:

```json
{"timestamp": "2025-01-20T14:30:22.123", "type": "transcription", "data": {"chunk_id": 1, "text": "你好", "language": "zh"}}
{"timestamp": "2025-01-20T14:30:22.456", "type": "translation", "data": {"chunk_id": 1, "source": "你好", "target": "Hello", "target_language": "en"}}
```

**Event Types**:
- `configuration`: Session configuration
- `transcription`: STT results
- `translation`: Translation results

### Metrics File (JSON)

Aggregated performance metrics:

```json
{
  "session_id": "20250120_143022",
  "start_time": "2025-01-20T14:30:22.000",
  "end_time": "2025-01-20T14:31:22.000",
  "total_chunks": 15,
  "chunks": [
    {
      "chunk_id": 1,
      "timestamp": "2025-01-20T14:30:22.123",
      "stt_latency_ms": 150.5,
      "translation_latency_ms": 45.2,
      "tts_latency_ms": 320.1,
      "total_latency_ms": 515.8
    }
  ],
  "averages": {
    "stt_latency_ms": 155.3,
    "translation_latency_ms": 42.8,
    "tts_latency_ms": 310.5,
    "total_latency_ms": 508.6
  }
}
```

## Analyzing Debug Output

Use the provided analysis script to generate reports:

```bash
# Analyze the most recent debug session
python examples/analyze_debug_output.py

# Analyze specific debug directory
python examples/analyze_debug_output.py ./my_debug
```

### Sample Analysis Output

```
📁 Analyzing debug session:
  Events: events_20250120_143022.jsonl
  Metrics: metrics_20250120_143022.json

======================================================================
📊 PERFORMANCE METRICS
======================================================================
Session ID: 20250120_143022
Start Time: 2025-01-20T14:30:22.000
End Time: 2025-01-20T14:31:22.000
Total Chunks: 15

Average Latencies:
  STT:         155.3 ms
  Translation: 42.8 ms
  TTS:         310.5 ms
  Total:       508.6 ms

Latency Range:
  Min: 450.2 ms
  Max: 620.5 ms

======================================================================
📝 TRANSLATION EVENTS
======================================================================
Total Transcriptions: 15
Total Translations: 15

Transcription → Translation Pairs:
----------------------------------------------------------------------

[1] Chunk 1
  Source (zh): 你好
  Target (en): Hello

[2] Chunk 2
  Source (zh): 今天天气很好
  Target (en): The weather is nice today

...
```

## Validation Checklist

Use debug output to verify:

### 1. Audio Quality ✓
- [ ] Listen to output WAV files
- [ ] Verify audio is clear and intelligible
- [ ] Check for distortion or artifacts
- [ ] Validate pronunciation accuracy

### 2. Translation Accuracy ✓
- [ ] Review transcription quality (STT accuracy)
- [ ] Check translation correctness
- [ ] Verify language detection
- [ ] Validate context preservation

### 3. Performance ✓
- [ ] Check average latencies meet requirements
- [ ] Identify bottlenecks (STT/Translation/TTS)
- [ ] Verify consistent performance across chunks
- [ ] Monitor for degradation over time

### 4. System Behavior ✓
- [ ] Verify chunk processing order
- [ ] Check for dropped chunks
- [ ] Validate error handling
- [ ] Monitor resource usage

## Common Issues and Solutions

### No Audio Output Files

**Problem**: `--save-audio` not specified or pipeline result has no audio

**Solution**:
```bash
# Ensure --save-audio flag is used
python examples/microphone_to_speaker.py --debug --save-audio
```

### Missing Transcription/Translation in Events

**Problem**: Pipeline result objects may not expose intermediate results

**Solution**: This is expected if pipeline only returns final audio. Check pipeline implementation for intermediate result access.

### High Latency

**Problem**: Total latency exceeds requirements

**Solution**:
1. Check metrics to identify bottleneck (STT/Translation/TTS)
2. Review provider configurations
3. Consider using faster models or GPU acceleration
4. Verify system resources (CPU/Memory/GPU)

### Audio Quality Issues

**Problem**: Output audio has distortion or poor quality

**Solution**:
1. Check sample rate consistency (16kHz)
2. Verify audio format conversion (float32 → int16)
3. Review TTS provider settings
4. Test with different TTS voices/models

## Advanced Usage

### Continuous Monitoring

```bash
# Run multiple sessions and compare results
for i in {1..5}; do
    python examples/microphone_to_speaker.py \
        --debug --save-audio \
        --duration 30 \
        --debug-dir "./debug_run_$i"

    python examples/analyze_debug_output.py "./debug_run_$i"
done
```

### Integration Testing

```python
import json
from pathlib import Path

def validate_translation_quality(debug_dir: Path, min_chunks: int = 10):
    """Validate translation session quality"""
    metrics_file = sorted((debug_dir / "metrics").glob("*.json"))[-1]

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Validation checks
    assert metrics["total_chunks"] >= min_chunks, "Insufficient chunks processed"
    assert metrics["averages"]["total_latency_ms"] < 1000, "Latency too high"

    # Check for consistent performance
    latencies = [c["total_latency_ms"] for c in metrics["chunks"]]
    assert max(latencies) - min(latencies) < 500, "Latency variance too high"

    print("✅ Translation quality validation passed")

# Usage
validate_translation_quality(Path("./debug_output"))
```

## Tips for Effective Debugging

1. **Start Simple**: Begin with short duration (10-30 seconds) to quickly iterate
2. **Save Audio**: Always use `--save-audio` for quality validation
3. **Consistent Testing**: Use same input for reproducible results
4. **Monitor Resources**: Watch CPU/GPU/Memory during runs
5. **Compare Sessions**: Run multiple sessions to identify patterns
6. **Review Logs**: Check event timeline for unexpected behavior

## See Also

- [Audio I/O Documentation](AUDIO_IO_SUMMARY.md)
- [Installation Guide](INSTALLATION.md)
- [Project Status](../PROJECT_STATUS.md)
