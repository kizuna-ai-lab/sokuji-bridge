"""
Gateway Service - FastAPI Server

Orchestrates STT, Translation, and TTS services via gRPC.
Exposes REST/WebSocket API for external clients.
"""

import asyncio
import base64
import io
import os
import struct
import sys
import tempfile
import time
import wave
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from services.gateway.grpc_clients import GRPCClients
from src.generated import common_pb2


# FastAPI app
app = FastAPI(
    title="Sokuji-Bridge Gateway",
    description="Real-time voice translation gateway service",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global gRPC clients
clients: Optional[GRPCClients] = None

# === Create API Routers ===
stt_router = APIRouter(prefix="/api/v1/stt", tags=["STT Service"])
translation_router = APIRouter(prefix="/api/v1/translation", tags=["Translation Service"])
tts_router = APIRouter(prefix="/api/v1/tts", tags=["TTS Service"])
pipeline_router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline Services"])
ws_router = APIRouter(prefix="/api/v1/ws", tags=["WebSocket"])


# Request/Response Models
class TranslateRequest(BaseModel):
    text: str
    source_language: str
    target_language: str
    voice_id: str = "default"


class TranslateResponse(BaseModel):
    transcription: str
    transcription_language: str
    translation: str
    translation_language: str
    audio_duration_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    services: dict


class AudioTranslateResponse(BaseModel):
    transcription: str
    transcription_language: str
    translation: str
    translation_language: str
    audio_data: str  # base64 encoded audio
    audio_format: str
    audio_sample_rate: int
    audio_duration_ms: float
    total_latency_ms: float


# === STT Models ===
class STTResponse(BaseModel):
    text: str
    language: str
    confidence: float
    processing_time_ms: float


# === Translation Models ===
class SimpleTranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


class SimpleTranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    processing_time_ms: float


# === TTS Models ===
class TTSRequest(BaseModel):
    text: str
    voice_id: str = Field(default="default", description="Voice ID for synthesis")
    language: str
    return_format: str = Field(default="json", description="Return format: 'json' or 'audio'")


class TTSResponse(BaseModel):
    audio_data: str  # base64 encoded
    audio_format: str
    sample_rate: int
    duration_ms: float
    processing_time_ms: float


# ============================================================================
# === Audio Conversion Utilities ===
# ============================================================================

def pcm_to_wav(pcm_data: bytes, sample_rate: int, num_channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Convert raw PCM audio data to WAV format

    Args:
        pcm_data: Raw PCM audio bytes (int16 format)
        sample_rate: Sample rate in Hz (e.g., 22050, 44100)
        num_channels: Number of audio channels (1=mono, 2=stereo)
        sample_width: Bytes per sample (2 for int16, 4 for int32)

    Returns:
        WAV formatted audio bytes with proper header
    """
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return wav_buffer.getvalue()


@app.on_event("startup")
async def startup_event():
    """Initialize gRPC clients on startup"""
    global clients

    # Get service URLs from environment
    stt_url = os.getenv("STT_SERVICE_URL", "localhost:50051")
    translation_url = os.getenv("TRANSLATION_SERVICE_URL", "localhost:50052")
    tts_url = os.getenv("TTS_SERVICE_URL", "localhost:50053")

    logger.info("Initializing gateway service...")
    logger.info(f"STT service: {stt_url}")
    logger.info(f"Translation service: {translation_url}")
    logger.info(f"TTS service: {tts_url}")

    clients = GRPCClients(stt_url, translation_url, tts_url)
    await clients.connect_all()

    logger.info("Gateway service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup gRPC clients on shutdown"""
    global clients
    if clients:
        await clients.disconnect_all()
        logger.info("Gateway service shut down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sokuji-Bridge Gateway API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/v1/info")
async def api_info():
    """Get detailed API information"""
    return {
        "name": "Sokuji-Bridge Gateway API",
        "version": "1.0.0",
        "description": "Real-time voice translation gateway with modular microservices",
        "endpoints": {
            "stt": {
                "base_path": "/api/v1/stt",
                "endpoints": [
                    "POST /transcribe - Transcribe audio to text",
                    "GET /languages - Get supported languages"
                ]
            },
            "translation": {
                "base_path": "/api/v1/translation",
                "endpoints": [
                    "POST /translate - Translate text",
                    "GET /languages - Get supported languages"
                ]
            },
            "tts": {
                "base_path": "/api/v1/tts",
                "endpoints": [
                    "POST /synthesize - Synthesize speech from text",
                    "GET /voices - Get available voices"
                ]
            },
            "pipeline": {
                "base_path": "/api/v1/pipeline",
                "endpoints": [
                    "POST /translate-text - Full text translation pipeline",
                    "POST /translate-audio - Full audio translation pipeline"
                ]
            },
            "websocket": {
                "base_path": "/api/v1/ws",
                "endpoints": [
                    "WS /translate - Real-time streaming translation"
                ]
            }
        },
        "microservices": {
            "stt": "Speech-to-Text service (port 50051)",
            "translation": "Translation service (port 50052)",
            "tts": "Text-to-Speech service (port 50053)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        health = await clients.health_check_all()

        all_healthy = all(
            isinstance(h, common_pb2.HealthCheckResponse) and h.healthy
            for h in health.values()
        )

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            services={
                "stt": health["stt"].status if hasattr(health["stt"], "status") else "error",
                "translation": health["translation"].status if hasattr(health["translation"], "status") else "error",
                "tts": health["tts"].status if hasattr(health["tts"], "status") else "error",
            },
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# === Pipeline Service Endpoints ===
# ============================================================================

@pipeline_router.post("/translate-text", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Translate text end-to-end (for testing without audio input)

    This endpoint simulates the full pipeline: receives text directly,
    translates it, and synthesizes speech.
    """
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        start_time = time.time()

        # Translate
        translation_result = await clients.translation.translate(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
        )

        # Synthesize
        synthesis_result = await clients.tts.synthesize(
            text=translation_result.translated_text,
            voice_id=request.voice_id,
            language=request.target_language,
        )

        total_latency_ms = (time.time() - start_time) * 1000

        return TranslateResponse(
            transcription=request.text,
            transcription_language=request.source_language,
            translation=translation_result.translated_text,
            translation_language=request.target_language,
            audio_duration_ms=synthesis_result.duration_ms,
            total_latency_ms=total_latency_ms,
        )

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@pipeline_router.post("/translate-audio")
async def translate_audio(
    audio_file: UploadFile = File(..., description="Audio file to translate"),
    source_language: str = Form("auto", description="Source language code"),
    target_language: str = Form(..., description="Target language code"),
    voice_id: str = Form("default", description="TTS voice ID"),
    return_format: str = Form("json", description="Return format: 'json' or 'audio'"),
):
    """
    Translate audio file end-to-end

    Process:
    1. Receive audio file
    2. STT: Convert speech to text
    3. Translation: Translate text
    4. TTS: Convert translated text to speech
    5. Return result (JSON with base64 audio or direct audio stream)

    Args:
        audio_file: Audio file to translate (wav, mp3, m4a, etc.)
        source_language: Source language code (default: auto-detect)
        target_language: Target language code (required)
        voice_id: TTS voice ID (default: "default")
        return_format: Response format - "json" or "audio" (default: "json")

    Returns:
        AudioTranslateResponse with base64 audio if format="json"
        Audio file stream if format="audio"
    """
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    # Validate file size (max 10MB)
    max_file_size = 10 * 1024 * 1024  # 10MB
    if audio_file.size and audio_file.size > max_file_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

    # Validate return format
    if return_format not in ["json", "audio"]:
        raise HTTPException(status_code=400, detail="Invalid return_format. Must be 'json' or 'audio'")

    try:
        start_time = time.time()

        # Read audio file
        audio_data = await audio_file.read()
        logger.info(f"Received audio file: {audio_file.filename}, size: {len(audio_data)} bytes")

        # Process audio for STT (normalize to 16kHz mono)
        processed_audio = process_audio_file(audio_data)

        # Create AudioChunk for STT
        audio_chunk = common_pb2.AudioChunk(
            data=processed_audio,
            sample_rate=16000,
            timestamp=time.time(),
            channels=1,
            format="int16",
        )

        # STT: Transcribe audio
        logger.info("Starting STT transcription...")
        transcription = await clients.stt.transcribe(
            audio_chunk,
            language=source_language if source_language != "auto" else None,
        )

        if not transcription.text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio file")

        logger.info(f"Transcribed: {transcription.text[:100]}... (lang: {transcription.language})")

        # Translation: Translate text
        logger.info(f"Translating from {transcription.language} to {target_language}...")
        translation = await clients.translation.translate(
            text=transcription.text,
            source_lang=transcription.language,
            target_lang=target_language,
        )

        logger.info(f"Translated: {translation.translated_text[:100]}...")

        # TTS: Synthesize speech
        logger.info(f"Synthesizing speech with voice {voice_id}...")
        synthesis = await clients.tts.synthesize(
            text=translation.translated_text,
            voice_id=voice_id,
            language=target_language,
        )

        total_latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Translation completed in {total_latency_ms:.0f}ms")

        # Convert int16 PCM to WAV if needed
        audio_data = synthesis.audio_data
        audio_format = synthesis.format

        if audio_format == "int16":
            logger.debug("Converting int16 PCM to WAV format")
            audio_data = pcm_to_wav(
                pcm_data=audio_data,
                sample_rate=synthesis.sample_rate,
                num_channels=1,
                sample_width=2
            )
            audio_format = "wav"

        # Return based on format
        if return_format == "json":
            # Return JSON with base64 encoded audio
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            return AudioTranslateResponse(
                transcription=transcription.text,
                transcription_language=transcription.language,
                translation=translation.translated_text,
                translation_language=target_language,
                audio_data=audio_base64,
                audio_format=audio_format,
                audio_sample_rate=synthesis.sample_rate,
                audio_duration_ms=synthesis.duration_ms,
                total_latency_ms=total_latency_ms,
            )
        else:
            # Return audio file directly
            audio_io = io.BytesIO(audio_data)
            media_type = "audio/wav" if audio_format == "wav" else f"audio/{audio_format}"

            return StreamingResponse(
                audio_io,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename=translated_{audio_file.filename}",
                    "X-Transcription": transcription.text,
                    "X-Translation": translation.translated_text,
                    "X-Processing-Time-Ms": str(total_latency_ms),
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# ============================================================================
# === WebSocket Endpoints ===
# ============================================================================

@ws_router.websocket("/translate")
async def websocket_translate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time translation

    Client sends audio chunks, receives translated audio back.
    Protocol:
    - Client sends: JSON config + binary audio chunks
    - Server sends: Binary audio chunks (synthesized translation)
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    if not clients:
        await websocket.close(code=1011, reason="Gateway not initialized")
        return

    try:
        # Receive initial configuration
        config = await websocket.receive_json()
        source_lang = config.get("source_language", "auto")
        target_lang = config.get("target_language", "en")
        voice_id = config.get("voice_id", "default")
        sample_rate = config.get("sample_rate", 16000)

        logger.info(f"WebSocket config: {source_lang} -> {target_lang}, voice={voice_id}")

        # Process audio stream
        while True:
            # Receive audio chunk
            audio_data = await websocket.receive_bytes()

            # Create AudioChunk proto
            audio_chunk = common_pb2.AudioChunk(
                data=audio_data,
                sample_rate=sample_rate,
                timestamp=time.time(),
                channels=1,
                format="int16",
            )

            # STT: Transcribe
            transcription = await clients.stt.transcribe(
                audio_chunk,
                language=source_lang if source_lang != "auto" else None,
            )

            # Skip if empty transcription
            if not transcription.text.strip():
                continue

            logger.debug(f"Transcribed: {transcription.text}")

            # Translation: Translate
            translation = await clients.translation.translate(
                text=transcription.text,
                source_lang=transcription.language,
                target_lang=target_lang,
            )

            logger.debug(f"Translated: {translation.translated_text}")

            # TTS: Synthesize
            synthesis = await clients.tts.synthesize(
                text=translation.translated_text,
                voice_id=voice_id,
                language=target_lang,
            )

            # Send synthesized audio back
            await websocket.send_bytes(synthesis.audio_data)

            # Optionally send metadata
            await websocket.send_json({
                "transcription": transcription.text,
                "translation": translation.translated_text,
                "duration_ms": synthesis.duration_ms,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))




# Audio processing helper functions
def process_audio_file(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """
    Process and normalize audio data for STT
    Convert to mono, 16kHz, int16 format using pydub
    """
    try:
        from pydub import AudioSegment
        import io

        # Load audio from bytes
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to target sample rate
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Convert to 16-bit PCM
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bits

        # Get raw audio data
        return audio.raw_data

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")


async def audio_bytes_to_chunks(audio_data: bytes, sample_rate: int = 16000, chunk_size_ms: int = 100) -> list:
    """
    Convert audio bytes to chunks for streaming
    """
    # Calculate chunk size in bytes (assuming int16 format)
    bytes_per_sample = 2  # int16
    samples_per_chunk = int(sample_rate * chunk_size_ms / 1000)
    chunk_size_bytes = samples_per_chunk * bytes_per_sample

    chunks = []
    for i in range(0, len(audio_data), chunk_size_bytes):
        chunk_data = audio_data[i:i+chunk_size_bytes]
        if len(chunk_data) > 0:
            chunks.append(chunk_data)

    return chunks


# ============================================================================
# === STT Service Endpoints ===
# ============================================================================

@stt_router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("auto", description="Language hint for recognition")
):
    """
    Transcribe audio to text using STT service

    This endpoint only performs speech-to-text conversion without translation.
    """
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        start_time = time.time()

        # Read audio file
        audio_data = await audio_file.read()
        logger.info(f"Received audio file for transcription: {audio_file.filename}, size: {len(audio_data)} bytes")

        # Process audio
        processed_audio = process_audio_file(audio_data)

        # Create AudioChunk
        audio_chunk = common_pb2.AudioChunk(
            data=processed_audio,
            sample_rate=16000,
            timestamp=time.time(),
            channels=1,
            format="int16",
        )

        # Transcribe
        transcription = await clients.stt.transcribe(
            audio_chunk,
            language=language if language != "auto" else None,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        return STTResponse(
            text=transcription.text,
            language=transcription.language,
            confidence=transcription.confidence,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@stt_router.get("/languages")
async def get_stt_languages():
    """Get supported languages for STT service"""
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        response = await clients.stt.stub.GetSupportedLanguages(common_pb2.Empty())
        return {"languages": list(response.language_codes)}
    except Exception as e:
        logger.error(f"Error getting STT languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# === Translation Service Endpoints ===
# ============================================================================

@translation_router.post("/translate", response_model=SimpleTranslationResponse)
async def translate_text_only(request: SimpleTranslationRequest):
    """
    Translate text using Translation service

    This endpoint only performs text translation without TTS.
    """
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        start_time = time.time()

        # Translate
        translation_result = await clients.translation.translate(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        return SimpleTranslationResponse(
            original_text=request.text,
            translated_text=translation_result.translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@translation_router.get("/languages")
async def get_translation_languages():
    """Get supported languages for Translation service"""
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        response = await clients.translation.stub.GetSupportedLanguages(common_pb2.Empty())
        return {
            "languages": list(response.language_codes),
            "supports_batch": response.supports_batch,
            "supports_streaming": response.supports_streaming,
        }
    except Exception as e:
        logger.error(f"Error getting Translation languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# === TTS Service Endpoints ===
# ============================================================================

@tts_router.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text using TTS service

    This endpoint only performs text-to-speech conversion.
    """
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        start_time = time.time()

        # Synthesize
        synthesis_result = await clients.tts.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Convert int16 PCM to WAV if needed
        audio_data = synthesis_result.audio_data
        audio_format = synthesis_result.format

        if audio_format == "int16":
            logger.debug("Converting int16 PCM to WAV format")
            audio_data = pcm_to_wav(
                pcm_data=audio_data,
                sample_rate=synthesis_result.sample_rate,
                num_channels=1,
                sample_width=2
            )
            audio_format = "wav"

        # Return based on format
        if request.return_format == "json":
            # Return JSON with base64 encoded audio
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            return TTSResponse(
                audio_data=audio_base64,
                audio_format=audio_format,
                sample_rate=synthesis_result.sample_rate,
                duration_ms=synthesis_result.duration_ms,
                processing_time_ms=processing_time_ms
            )
        else:
            # Return audio file directly
            audio_io = io.BytesIO(audio_data)
            media_type = "audio/wav" if audio_format == "wav" else f"audio/{audio_format}"

            return StreamingResponse(
                audio_io,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename=synthesized.wav",
                    "X-Duration-Ms": str(synthesis_result.duration_ms),
                    "X-Processing-Time-Ms": str(processing_time_ms),
                }
            )

    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@tts_router.get("/voices")
async def get_tts_voices():
    """Get available voices for TTS service"""
    if not clients:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        response = await clients.tts.stub.GetVoices(common_pb2.Empty())
        voices = []
        for voice in response.voices:
            voices.append({
                "id": voice.id,
                "name": voice.name,
                "language": voice.language,
                "supported_languages": list(voice.supported_languages),
                "gender": voice.gender,
            })
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting TTS voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# Include all routers in the main app
app.include_router(stt_router)
app.include_router(translation_router)
app.include_router(tts_router)
app.include_router(pipeline_router)
app.include_router(ws_router)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    # Run server
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=None,  # Use loguru instead of uvicorn logging
    )
