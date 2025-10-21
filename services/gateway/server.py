"""
Gateway Service - FastAPI Server

Orchestrates STT, Translation, and TTS services via gRPC.
Exposes REST/WebSocket API for external clients.
"""

import asyncio
import os
import sys
import time
from typing import AsyncIterator, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    return {"message": "Sokuji-Bridge Gateway API", "version": "0.1.0"}


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


@app.post("/translate/text", response_model=TranslateResponse)
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


@app.websocket("/ws/translate")
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


@app.get("/services/stt/languages")
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


@app.get("/services/translation/languages")
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


@app.get("/services/tts/voices")
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
