# Audio I/O Documentation

Complete guide to using Sokuji-Bridge audio input/output system.

## Overview

The Audio I/O system provides a unified interface for capturing and playing audio through various sources:

- **Microphone Input**: Real-time audio capture from system microphones
- **Speaker Output**: Real-time audio playback through system speakers
- **File I/O**: Read and write audio files in multiple formats
- **VAD**: Voice Activity Detection for intelligent audio segmentation

## Quick Start

### Real-time Translation (Microphone → Speaker)

```python
from utils import MicrophoneInput, SpeakerOutput

# Initialize audio I/O
mic = MicrophoneInput(sample_rate=16000, channels=1)
speaker = SpeakerOutput(sample_rate=16000, channels=1)

await mic.start()
await speaker.start()

# Process through pipeline
async for result in pipeline.process_audio_stream(mic.stream()):
    await speaker.play(result)

await speaker.stop()
await mic.stop()
```

### File Translation

```python
from utils import AudioFileReader, AudioFileWriter

# Read audio file
reader = AudioFileReader("input.wav", chunk_duration_ms=1000)
await reader.start()

# Write translated audio
writer = AudioFileWriter("output.wav", sample_rate=16000)
await writer.start()

# Process
async for result in pipeline.process_audio_stream(reader.stream()):
    await writer.play(result)

await writer.stop()
await reader.stop()
```

## API Reference

### AudioInput (Abstract Base Class)

Base class for all audio input sources.

#### Methods

- `async start()` - Start audio input stream
- `async stop()` - Stop audio input stream
- `async stream() -> AsyncIterator[AudioChunk]` - Stream audio chunks
- `@property is_running -> bool` - Check if input is running
- `@staticmethod get_devices() -> List[AudioDevice]` - List available devices

#### Context Manager Support

```python
async with MicrophoneInput() as mic:
    async for chunk in mic.stream():
        # Process audio
        pass
```

### MicrophoneInput

Real-time microphone audio capture.

#### Constructor

```python
MicrophoneInput(
    device: Optional[int] = None,      # Device index (None = default)
    sample_rate: int = 16000,          # Sample rate in Hz
    channels: int = 1,                 # Number of channels
    chunk_size: int = 1024,            # Samples per chunk
    dtype: str = "int16"               # Audio format
)
```

#### Example

```python
# List available microphones
devices = MicrophoneInput.get_devices()
for device in devices:
    print(f"[{device.index}] {device.name}")

# Use specific microphone
mic = MicrophoneInput(device=0, sample_rate=16000)
await mic.start()

# Capture for 5 seconds
import time
start = time.time()
async for chunk in mic.stream():
    print(f"Captured {chunk.duration_ms}ms")
    if time.time() - start > 5:
        break

await mic.stop()
```

### AudioOutput (Abstract Base Class)

Base class for all audio output destinations.

#### Methods

- `async start()` - Start audio output stream
- `async stop()` - Stop audio output and flush buffers
- `async play(audio: SynthesisResult)` - Play single audio chunk
- `async play_stream(audio_stream: AsyncIterator[SynthesisResult])` - Play audio stream
- `@property is_running -> bool` - Check if output is running
- `@staticmethod get_devices() -> List[AudioDevice]` - List available devices

### SpeakerOutput

Real-time speaker audio playback.

#### Constructor

```python
SpeakerOutput(
    device: Optional[int] = None,      # Device index (None = default)
    sample_rate: int = 16000,          # Sample rate in Hz
    channels: int = 1,                 # Number of channels
    buffer_size: int = 2048            # Output buffer size
)
```

#### Example

```python
# List available speakers
devices = SpeakerOutput.get_devices()
for device in devices:
    print(f"[{device.index}] {device.name}")

# Play audio
speaker = SpeakerOutput(device=0)
await speaker.start()

# Play single audio chunk
await speaker.play(synthesis_result)

# Or play stream
async for result in pipeline.process_audio_stream(input_stream):
    await speaker.play(result)

await speaker.stop()
```

### AudioFileReader

Read audio files and stream as AudioChunk objects.

#### Constructor

```python
AudioFileReader(
    file_path: str,                           # Path to audio file
    chunk_duration_ms: float = 1000.0,        # Chunk duration in ms
    target_sample_rate: Optional[int] = None, # Convert to sample rate
    target_channels: Optional[int] = None     # Convert to channels
)
```

#### Supported Formats

- WAV (Waveform Audio File Format)
- FLAC (Free Lossless Audio Codec)
- OGG (Ogg Vorbis)

#### Example

```python
# Read audio file
reader = AudioFileReader(
    "audio.wav",
    chunk_duration_ms=1000,
    target_sample_rate=16000,
    target_channels=1
)

await reader.start()

# Get file info
info = reader.get_info()
print(f"Duration: {info['duration_ms']:.1f}ms")
print(f"Sample rate: {info['sample_rate']}Hz")
print(f"Channels: {info['channels']}")

# Stream chunks
async for chunk in reader.stream():
    # Process chunk
    print(f"Read {chunk.duration_ms}ms")

await reader.stop()
```

### AudioFileWriter

Write audio to file.

#### Constructor

```python
AudioFileWriter(
    file_path: str,              # Output file path
    sample_rate: int = 16000,    # Sample rate in Hz
    channels: int = 1,           # Number of channels
    format: str = "WAV",         # Audio format
    subtype: str = "PCM_16"      # Audio subtype
)
```

#### Example

```python
# Write audio file
writer = AudioFileWriter(
    "output.wav",
    sample_rate=16000,
    channels=1
)

await writer.start()

# Write single audio
await writer.play(synthesis_result)

# Or write stream
async for result in pipeline.process_audio_stream(input_stream):
    await writer.play(result)

await writer.stop()

# Check output
info = writer.get_info()
print(f"Wrote {info['frames_written']} frames")
print(f"Duration: {info['duration_ms']:.1f}ms")
```

## Voice Activity Detection (VAD)

VAD detects speech in audio and enables intelligent segmentation.

### VADProvider (Abstract Base Class)

Base class for VAD implementations.

#### Methods

- `async initialize()` - Initialize VAD
- `async cleanup()` - Clean up VAD resources
- `async detect_speech(audio: AudioChunk) -> VADResult` - Detect speech
- `async segment_audio(audio_stream, ...) -> AsyncIterator[AudioChunk]` - Segment audio

### DummyVAD

Always detects speech (for testing).

```python
vad = DummyVAD()
await vad.initialize()

result = await vad.detect_speech(audio_chunk)
print(f"Speech detected: {result.is_speech}")  # Always True

await vad.cleanup()
```

### EnergyVAD

Energy-based VAD (simple but fast).

```python
vad = EnergyVAD(energy_threshold=0.05)
await vad.initialize()

# Detect speech in single chunk
result = await vad.detect_speech(audio_chunk)
print(f"Speech: {result.is_speech}, Confidence: {result.confidence}")

# Segment audio stream
async for segment in vad.segment_audio(
    mic.stream(),
    min_speech_duration_ms=250,
    min_silence_duration_ms=500
):
    # Each segment contains complete speech utterance
    await pipeline.process_single(segment)

await vad.cleanup()
```

## Device Management

### List Audio Devices

```python
from utils import list_audio_devices

# List all devices
devices = list_audio_devices()
for device in devices:
    print(device)

# List only input devices
input_devices = list_audio_devices(input_only=True)

# List only output devices
output_devices = list_audio_devices(output_only=True)
```

### Get Default Devices

```python
from utils import get_default_input_device, get_default_output_device

# Get default microphone
mic_device = get_default_input_device()
print(f"Default microphone: {mic_device.name}")

# Get default speaker
speaker_device = get_default_output_device()
print(f"Default speaker: {speaker_device.name}")
```

## Complete Examples

### Example 1: Real-time Translation

See `examples/microphone_to_speaker.py` for a complete real-time translation example.

```bash
python examples/microphone_to_speaker.py --source zh --target en --duration 30
```

### Example 2: File Translation

See `examples/file_translation.py` for file translation examples.

```bash
# Single file
python examples/file_translation.py input.wav -o output.wav --source zh --target en

# Batch processing
python examples/file_translation.py audio1.wav audio2.wav audio3.wav -o translated/
```

## Performance Considerations

### Buffer Sizes

- **Microphone chunk_size**: Smaller = lower latency, higher CPU usage
- **Speaker buffer_size**: Larger = smoother playback, higher latency

Recommended values:
- Low latency: chunk_size=512, buffer_size=1024
- Balanced: chunk_size=1024, buffer_size=2048 (default)
- High quality: chunk_size=2048, buffer_size=4096

### Sample Rates

- **16000 Hz**: Recommended for speech (optimal for STT models)
- **22050 Hz**: Good balance for speech and music
- **44100 Hz**: CD quality (higher CPU and memory usage)
- **48000 Hz**: Professional audio

### File I/O

- Use `chunk_duration_ms=1000` for balanced memory usage
- Smaller chunks = lower memory, more overhead
- Larger chunks = higher memory, less overhead

## Troubleshooting

### No Audio Devices Found

```python
devices = list_audio_devices()
if len(devices) == 0:
    print("No audio devices found!")
    # Check system audio drivers
```

### Microphone Not Working

1. Check device permissions (Linux/macOS require microphone access)
2. List devices and verify device index
3. Test with system audio settings

### Speaker Playback Issues

1. Verify sample rate matches output device
2. Check speaker volume and mute settings
3. Try different buffer sizes

### File Format Not Supported

Currently supported: WAV, FLAC, OGG

For MP3 support, install additional dependencies:
```bash
pip install pydub
```

## Advanced Usage

### Custom Audio Processing

```python
class CustomAudioInput(AudioInput):
    async def start(self):
        # Custom initialization
        self._is_running = True

    async def stop(self):
        self._is_running = False

    async def stream(self):
        while self._is_running:
            # Custom audio generation
            audio_data = generate_custom_audio()
            yield AudioChunk(
                data=audio_data,
                sample_rate=16000,
                timestamp=time.time(),
                channels=1,
                format="int16"
            )

    @staticmethod
    def get_devices():
        return []
```

### Audio Format Conversion

The system automatically handles:
- Sample rate conversion (resampling)
- Channel conversion (mono ↔ stereo)
- Format conversion (int16 ↔ float32)

## Future Enhancements

Planned features:
- Silero VAD integration (ML-based speech detection)
- WebRTC VAD support
- MP3 file support
- Network audio streaming (RTP/RTSP)
- Audio effects and filters

## See Also

- [Provider Guide](./providers.md) - STT, Translation, TTS providers
- [Pipeline Guide](./pipeline.md) - Translation pipeline usage
- [Configuration](./configuration.md) - System configuration
- [Examples](../examples/) - Complete usage examples
