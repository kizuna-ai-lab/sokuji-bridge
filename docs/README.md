# Sokuji-Bridge Documentation

Complete documentation for the Sokuji-Bridge real-time speech-to-speech translation system.

## Quick Navigation

### Getting Started
- [Installation Guide](installation.md) - System requirements and setup
- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [Quick Start (Monolithic)](quickstart-monolithic.md) - Single-process deployment

### Core Documentation
- [Architecture Overview](architecture.md) - System design and components
- [Architecture (Microservices)](architecture-microservices.md) - Distributed deployment
- [API Reference](api.md) - REST and WebSocket APIs
- [Audio I/O](audio_io.md) - Audio input/output handling

### Provider Documentation
- [Providers Overview](providers.md) - STT, Translation, and TTS providers
- [Faster-Whisper VAD](providers/faster-whisper-vad.md) - Built-in VAD configuration

### Feature Guides

#### SimulStreaming (Real-Time STT)
- **[SimulStreaming Setup](features/simulstreaming-setup.md)** - Installation and configuration
- **[SimulStreaming + VAD](features/simulstreaming-vad.md)** - Automatic sentence segmentation
  - Quick start and configuration
  - Scenario-based tuning
  - Installation and troubleshooting
  - Complete usage examples

### User Guides
- [Model Selection Guide](guides/model-selection.md) - Choose the right models
- [Microphone Real-Time Transcription](guides/microphone-realtime.md) - Live audio processing
- [Debug Mode](guides/debug-mode.md) - Debugging configuration
- [Debug Usage](guides/debug-usage.md) - Debugging techniques

### Advanced Topics
- [Performance Optimization](performance.md) - Tuning for production
- [Development Guide](development.md) - Contributing and development setup

## Documentation Structure

```
docs/
├── README.md                          # This file
├── installation.md                    # Setup guide
├── quickstart.md                      # 5-minute guide
├── architecture.md                    # System architecture
├── api.md                             # API reference
├── providers.md                       # Provider overview
│
├── features/                          # Feature-specific guides
│   ├── simulstreaming-setup.md        # SimulStreaming installation
│   └── simulstreaming-vad.md          # VAD automatic segmentation ⭐
│
├── guides/                            # User guides
│   ├── model-selection.md             # Model selection
│   ├── microphone-realtime.md         # Real-time audio
│   ├── debug-mode.md                  # Debug configuration
│   └── debug-usage.md                 # Debug techniques
│
└── providers/                         # Provider-specific docs
    └── faster-whisper-vad.md          # Faster-Whisper VAD config
```

## Key Features

### Real-Time Speech Processing
- **SimulStreaming**: Low-latency (<2s) streaming transcription
- **VAD Integration**: Automatic sentence segmentation
- **Multi-Language**: Support for 20+ languages
- **GPU Acceleration**: CUDA-optimized processing

### Translation Pipeline
- **NLLB**: High-quality neural translation
- **Context Preservation**: Maintains meaning across sentences
- **Batch Processing**: Efficient multi-sentence handling

### Text-to-Speech
- **Piper TTS**: Fast, natural-sounding synthesis
- **Voice Selection**: Multiple voices per language
- **Quality Tuning**: Adjustable speed, noise, and pitch

## Popular Topics

### For Beginners
1. [Installation Guide](installation.md)
2. [Quick Start](quickstart.md)
3. [Model Selection](guides/model-selection.md)

### For Production
1. [Architecture (Microservices)](architecture-microservices.md)
2. [Performance Optimization](performance.md)
3. [SimulStreaming + VAD](features/simulstreaming-vad.md)

### For Developers
1. [Development Guide](development.md)
2. [API Reference](api.md)
3. [Providers Overview](providers.md)

## Examples

Located in the `examples/` directory:
- `microphone_simulstreaming.py` - Real-time transcription (with optional --vad flag)
- `compare_stt_providers.py` - Provider comparison

Usage:
```bash
# Basic mode
python examples/microphone_simulstreaming.py

# With VAD
python examples/microphone_simulstreaming.py --vad
```

## Getting Help

### Documentation Issues
- Missing information? [Open an issue](https://github.com/kizunaai/sokuji-bridge/issues)
- Found an error? Submit a pull request

### Technical Support
1. Check the relevant guide above
2. Review [troubleshooting sections](features/simulstreaming-vad.md#troubleshooting)
3. Search existing [GitHub issues](https://github.com/kizunaai/sokuji-bridge/issues)
4. Open a new issue with:
   - System information
   - Configuration files
   - Error logs
   - Steps to reproduce

## Contributing

See [Development Guide](development.md) for:
- Development environment setup
- Code standards and style
- Testing requirements
- Pull request process

## License

This documentation is part of the Sokuji-Bridge project.
See the main [LICENSE](../LICENSE) file for details.
