# Monolith Mode (Legacy)

This directory is reserved for future single-process (monolithic) application mode.

## Purpose

While the current architecture uses microservices (`/services/` directory), this directory can be used for:

1. **Single-process deployment**: Run all providers in a single Python process
2. **Development mode**: Easier debugging without Docker
3. **Resource-constrained environments**: When you cannot run multiple Docker containers
4. **Legacy compatibility**: Maintain backward compatibility with older deployment methods

## Current Status

**Currently empty** - The project uses microservices architecture by default.

## Microservices vs Monolith

### Microservices (Current - `/services/`)
- **Location**: Top-level `services/` directory
- **Architecture**: STT, Translation, TTS, Gateway as separate gRPC services
- **Deployment**: Docker Compose with 4 containers
- **Benefits**: Dependency isolation, independent scaling, provider flexibility
- **Use when**: Production deployment, multiple providers, GPU isolation needed

### Monolith (Future - `/src/monolith/`)
- **Location**: This directory (`src/monolith/`)
- **Architecture**: All providers in a single Python process
- **Deployment**: Single Python application or single Docker container
- **Benefits**: Simpler deployment, lower resource usage, easier debugging
- **Use when**: Development, testing, resource-constrained environments

## Migration Path

If you need to implement monolith mode in the future:

1. **Keep existing code**:
   - `src/providers/` - Provider implementations (shared)
   - `src/core/pipeline.py` - Pipeline orchestrator (works in both modes)
   - `src/utils/` - Audio utilities (shared)

2. **Add monolith-specific code here**:
   ```
   src/monolith/
   ├── __init__.py
   ├── main.py          # Entry point for monolith mode
   ├── api.py           # FastAPI/Flask REST API
   └── cli.py           # Command-line interface
   ```

3. **Example monolith main.py**:
   ```python
   # src/monolith/main.py
   from src.core.pipeline import TranslationPipeline
   from src.providers.stt.faster_whisper_provider import FasterWhisperProvider
   from src.providers.translation.nllb_provider import NLLBProvider
   from src.providers.tts.piper_provider import PiperProvider

   async def main():
       # Initialize providers
       stt = FasterWhisperProvider(config)
       translator = NLLBProvider(config)
       tts = PiperProvider(config)

       # Create pipeline
       pipeline = TranslationPipeline(stt, translator, tts, config)
       await pipeline.initialize()

       # Process audio
       result = await pipeline.process_single(audio_chunk)
   ```

## Why This Directory Was Renamed

This directory was originally named `src/services/` which caused confusion with the microservices architecture in the top-level `services/` directory.

**Naming clarification**:
- `/services/` (top-level) = **Microservices** (gRPC servers, Docker containers)
- `/src/monolith/` (this directory) = **Monolithic mode** (single-process alternative)
- `/src/providers/` = **Provider implementations** (shared by both architectures)
- `/src/core/` = **Core logic** (shared by both architectures)

## See Also

- `QUICKSTART_MICROSERVICES.md` - Microservices deployment guide
- `docs/MICROSERVICES.md` - Microservices architecture documentation
- `src/core/pipeline.py` - Pipeline orchestrator (works in both modes)
