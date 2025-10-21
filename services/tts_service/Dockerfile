# TTS Service Dockerfile - Multi-stage build supporting multiple providers
# Build with: docker build --build-arg PROVIDER=piper -t tts-service .

# Declare ARG at global scope (before any FROM) so it can be used in all stages
ARG PROVIDER=piper

# ============================================================================
# Stage 1: Base image with common dependencies (for CPU providers)
# ============================================================================
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install base Python dependencies (shared across all providers)
COPY services/tts_service/requirements/base.txt /tmp/base.txt
RUN pip install --no-cache-dir -r /tmp/base.txt

# ============================================================================
# Stage 2: Piper provider (CPU-optimized, fast and lightweight)
# ============================================================================
FROM base AS provider-piper

# Install Piper dependencies
COPY services/tts_service/requirements/piper.txt /tmp/piper.txt
RUN pip install --no-cache-dir -r /tmp/piper.txt

# ============================================================================
# Stage 3: XTTS provider (GPU-accelerated, high quality)
# Uses CUDA base image for GPU support
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS xtts-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install base and XTTS dependencies
COPY services/tts_service/requirements/base.txt /tmp/base.txt
COPY services/tts_service/requirements/xtts.txt /tmp/xtts.txt
RUN pip3 install --no-cache-dir -r /tmp/base.txt -r /tmp/xtts.txt

FROM xtts-base AS provider-xtts

# ============================================================================
# Stage 4: ElevenLabs provider (API-based, lightweight)
# ============================================================================
FROM base AS provider-elevenlabs

# Install ElevenLabs API dependencies
COPY services/tts_service/requirements/elevenlabs.txt /tmp/elevenlabs.txt
RUN pip install --no-cache-dir -r /tmp/elevenlabs.txt

# ============================================================================
# Stage 5: Kokoro provider (experimental, local)
# ============================================================================
FROM base AS provider-kokoro

# Install Kokoro dependencies
COPY services/tts_service/requirements/kokoro.txt /tmp/kokoro.txt
RUN pip install --no-cache-dir -r /tmp/kokoro.txt

# ============================================================================
# Final stage: Select provider based on build argument
# ============================================================================
FROM provider-${PROVIDER} AS final

# Copy project code
COPY src/ /app/src/
COPY services/tts_service/ /app/services/tts_service/

# Expose gRPC port
EXPOSE 50053

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PORT=50053

# For XTTS provider using CUDA base image, use python3 instead of python
CMD ["/bin/sh", "-c", "if command -v python >/dev/null 2>&1; then python /app/services/tts_service/server.py; else python3 /app/services/tts_service/server.py; fi"]
