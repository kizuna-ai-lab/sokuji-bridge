# STT Service Dockerfile - Multi-stage build supporting multiple providers
# Build with: docker build --build-arg PROVIDER=faster_whisper -t stt-service .

# Declare ARG at global scope (before any FROM) so it can be used in all stages
ARG PROVIDER=faster_whisper

# ============================================================================
# Stage 1: Base image with common dependencies
# ============================================================================
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install base Python dependencies (shared across all providers)
COPY services/stt_service/requirements/base.txt /tmp/base.txt
RUN pip install --no-cache-dir -r /tmp/base.txt

# ============================================================================
# Stage 2: Faster-Whisper provider (local, GPU-accelerated)
# ============================================================================
FROM base AS provider-faster_whisper

# Install Faster-Whisper dependencies
COPY services/stt_service/requirements/faster_whisper.txt /tmp/faster_whisper.txt
RUN pip install --no-cache-dir -r /tmp/faster_whisper.txt

# ============================================================================
# Stage 3: Whisper API provider (OpenAI API, lightweight)
# ============================================================================
FROM base AS provider-whisper_api

# Install OpenAI API dependencies
COPY services/stt_service/requirements/whisper_api.txt /tmp/whisper_api.txt
RUN pip install --no-cache-dir -r /tmp/whisper_api.txt

# ============================================================================
# Stage 4: Azure STT provider (Azure API, lightweight)
# ============================================================================
FROM base AS provider-azure_stt

# Install Azure Cognitive Services dependencies
COPY services/stt_service/requirements/azure_stt.txt /tmp/azure_stt.txt
RUN pip install --no-cache-dir -r /tmp/azure_stt.txt

# ============================================================================
# Final stage: Select provider based on build argument
# ============================================================================
FROM provider-${PROVIDER} AS final

# Copy project code
COPY src/ /app/src/
COPY services/stt_service/ /app/services/stt_service/

# Expose gRPC port
EXPOSE 50051

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PORT=50051

# Run the server
CMD ["python", "/app/services/stt_service/server.py"]
