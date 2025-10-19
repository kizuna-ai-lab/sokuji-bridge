# VAD Configuration Guide

Voice Activity Detection (VAD) configuration for faster-whisper in Sokuji-Bridge.

> **⚠️ Important Change (v0.2.0):**
> Pipeline VAD (`src/utils/vad.py`) has been removed as of v0.2.0.
> **All STT providers have built-in VAD** - faster-whisper uses **Silero VAD** internally.
> This document covers configuration of faster-whisper's built-in VAD parameters only.

## Overview

Sokuji-Bridge uses faster-whisper's built-in VAD (powered by Silero VAD) to filter out non-speech audio and improve transcription quality. All VAD parameters can be configured in `configs/default.yaml`.

### Why No Separate Pipeline VAD?

- **faster-whisper** includes high-quality Silero VAD (1.8MB, processes 30ms chunks in ~1ms)
- **OpenAI Whisper API** has server-side VAD
- **Azure Speech Services** has built-in VAD
- Separate pipeline VAD was redundant and caused confusion
- Direct STT provider VAD is more accurate and better integrated

## Configuration Location

VAD parameters are located in the STT configuration section:

```yaml
stt:
  provider: faster_whisper
  config:
    # Enable/disable VAD
    vad_filter: true

    # VAD Parameters
    vad_threshold: 0.95
    vad_neg_threshold: 0.35
    vad_speech_pad_ms: 400
    vad_min_speech_duration_ms: 250
    vad_max_speech_duration_s: 30.0
    vad_min_silence_duration_ms: 2000
```

## Parameters

### `vad_filter` (boolean)
- **Default:** `true`
- **Description:** Enable or disable VAD filtering
- **When to disable:** If you're experiencing over-aggressive filtering or working with non-standard audio

### `vad_threshold` (float, 0.0-1.0)
- **Default:** `0.95`
- **Description:** Probability threshold for speech detection. Higher = stricter filtering.
- **Range:** 0.0 (accept everything) to 1.0 (very strict)
- **Tuning:**
  - **Too high (>0.95):** May cut off speech, lose quiet utterances
  - **Too low (<0.5):** May include background noise, music, hallucinations
  - **Recommended:** 0.5-0.95 depending on audio quality

### `vad_neg_threshold` (float, 0.0-1.0)
- **Default:** `0.35`
- **Description:** Negative threshold for non-speech detection
- **Purpose:** Helps identify and filter out non-speech segments
- **Tuning:**
  - **Higher (>0.5):** More aggressive non-speech filtering
  - **Lower (<0.3):** More permissive, may include some non-speech

### `vad_speech_pad_ms` (integer, milliseconds)
- **Default:** `400`
- **Description:** Padding added before and after detected speech segments
- **Purpose:** Prevents cutting off the beginning/end of words
- **Tuning:**
  - **Too high (>800ms):** May include pre/post-speech noise
  - **Too low (<200ms):** May cut off word edges
  - **Recommended:** 300-600ms

### `vad_min_speech_duration_ms` (integer, milliseconds)
- **Default:** `250`
- **Description:** Minimum duration of speech to process
- **Purpose:** Filter out very short noises mistaken for speech
- **Tuning:**
  - **Higher (>500ms):** Filters more aggressively, may lose short utterances
  - **Lower (<100ms):** More permissive, may include noise bursts
  - **Recommended:** 100-500ms depending on speaking style

### `vad_max_speech_duration_s` (float, seconds)
- **Default:** `30.0`
- **Description:** Maximum duration of a single speech segment
- **Purpose:** Split long utterances for processing
- **Tuning:**
  - **Higher (>60s):** Allows longer continuous speech, more memory usage
  - **Lower (<10s):** More frequent splits, may interrupt sentences
  - **Recommended:** 20-60s for normal speech

### `vad_min_silence_duration_ms` (integer, milliseconds)
- **Default:** `2000`
- **Description:** Minimum silence duration to split speech segments
- **Purpose:** Determine when to end a speech segment
- **Tuning:**
  - **Higher (>3000ms):** Waits longer for pauses, combines more speech
  - **Lower (<500ms):** Splits more frequently, may break sentences
  - **Recommended:**
    - **Fast response:** 300-1000ms
    - **Complete sentences:** 1500-3000ms

## Common Scenarios

### Scenario 1: Noisy Environment
**Problem:** Background noise triggers false speech detection

**Solution:**
```yaml
vad_threshold: 0.8          # Stricter speech detection
vad_neg_threshold: 0.5      # More aggressive non-speech filtering
vad_min_speech_duration_ms: 400  # Ignore short noise bursts
```

### Scenario 2: Quiet Speaker
**Problem:** Soft-spoken audio gets filtered out

**Solution:**
```yaml
vad_threshold: 0.5          # More permissive detection
vad_speech_pad_ms: 600      # Extra padding to catch edges
vad_min_speech_duration_ms: 100  # Allow shorter utterances
```

### Scenario 3: Fast-Paced Conversation
**Problem:** Quick responses get cut off

**Solution:**
```yaml
vad_min_silence_duration_ms: 500  # Split faster
vad_speech_pad_ms: 300            # Less padding for speed
vad_max_speech_duration_s: 10.0   # Shorter segments
```

### Scenario 4: Music/Audio Hallucinations
**Problem:** Background music or audio triggers transcription

**Solution:**
```yaml
vad_threshold: 0.95         # Very strict speech detection
vad_neg_threshold: 0.35     # Default non-speech filtering
vad_min_speech_duration_ms: 500  # Ignore short sounds
```

### Scenario 5: Long Presentations/Lectures
**Problem:** Need to process long continuous speech

**Solution:**
```yaml
vad_max_speech_duration_s: 60.0     # Allow longer segments
vad_min_silence_duration_ms: 3000   # Wait for natural pauses
vad_threshold: 0.7                  # Moderate detection
```

## Performance Impact

### Memory Usage
- **Higher `vad_max_speech_duration_s`:** More memory needed to buffer audio
- **Recommendation:** Keep under 60s unless processing lectures

### Latency
- **Higher `vad_min_silence_duration_ms`:** Longer wait for transcription
- **Lower values:** Faster response but may split sentences
- **Trade-off:**
  - Real-time conversation: 300-1000ms
  - Quality transcription: 1500-3000ms

### Accuracy
- **Stricter VAD (`vad_threshold` > 0.8):** Fewer hallucinations, may lose speech
- **Permissive VAD (`vad_threshold` < 0.6):** More complete speech, may include noise

## Testing Your Configuration

Use the test script to validate VAD settings:

```bash
python examples/test_stt_only.py
```

Monitor the output for:
- ✅ Speech detected when you speak
- ✅ No transcription during silence
- ✅ No hallucinations from background sounds
- ✅ Complete words/sentences (not cut off)

## Migration from Pipeline VAD (v0.1.x → v0.2.0)

If you were using `pipeline.vad` configuration in v0.1.x:

```yaml
# OLD (v0.1.x) - No longer supported
pipeline:
  vad:
    enabled: true
    model: "silero"
    threshold: 0.5
```

```yaml
# NEW (v0.2.0+) - Use STT provider VAD
stt:
  config:
    vad_filter: true
    vad_threshold: 0.5  # Equivalent to old threshold
```

**For all use cases, STT provider's built-in VAD is sufficient and more accurate.**

## Troubleshooting

### Speech is cut off
- Increase `vad_speech_pad_ms` (try 600-800ms)
- Lower `vad_threshold` (try 0.6-0.7)
- Increase `vad_max_speech_duration_s`

### Too many hallucinations
- Increase `vad_threshold` (try 0.85-0.95)
- Increase `vad_min_speech_duration_ms` (try 400-600ms)
- Increase `vad_neg_threshold` (try 0.4-0.5)

### Transcription is slow
- Decrease `vad_min_silence_duration_ms` (try 500-1000ms)
- Decrease `vad_max_speech_duration_s` (try 10-20s)

### Words are broken across segments
- Increase `vad_min_silence_duration_ms` (try 2000-3000ms)
- Increase `vad_max_speech_duration_s` (try 30-60s)

## References

- [faster-whisper VAD documentation](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [VAD Best Practices](./VAD_FIX.md) - Sokuji-Bridge specific fixes

## See Also

- [Configuration Guide](./CONFIGURATION.md) - General configuration
- [Audio I/O Documentation](./audio_io.md) - Audio processing pipeline
- [DEBUG_USAGE.md](./DEBUG_USAGE.md) - Debugging VAD issues
