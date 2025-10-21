# Translation Service Dockerfile - Multi-stage build supporting multiple providers
# Build with: docker build --build-arg PROVIDER=nllb_local -t translation-service .

# Declare ARG at global scope (before any FROM) so it can be used in all stages
ARG PROVIDER=nllb_local

# ============================================================================
# Stage 1: Base image with common dependencies
# ============================================================================
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install base Python dependencies (shared across all providers)
COPY services/translation_service/requirements/base.txt /tmp/base.txt
RUN pip install --no-cache-dir -r /tmp/base.txt

# ============================================================================
# Stage 2: NLLB Local provider (local, GPU-accelerated)
# ============================================================================
FROM base AS provider-nllb_local

# Install NLLB dependencies
COPY services/translation_service/requirements/nllb_local.txt /tmp/nllb_local.txt
RUN pip install --no-cache-dir -r /tmp/nllb_local.txt

# ============================================================================
# Stage 3: DeepL API provider (cloud, lightweight)
# ============================================================================
FROM base AS provider-deepl_api

# Install DeepL API dependencies
COPY services/translation_service/requirements/deepl_api.txt /tmp/deepl_api.txt
RUN pip install --no-cache-dir -r /tmp/deepl_api.txt

# ============================================================================
# Stage 4: OpenAI GPT-4 provider (cloud, lightweight)
# ============================================================================
FROM base AS provider-openai_gpt4

# Install OpenAI API dependencies
COPY services/translation_service/requirements/openai_gpt4.txt /tmp/openai_gpt4.txt
RUN pip install --no-cache-dir -r /tmp/openai_gpt4.txt

# ============================================================================
# Stage 5: Google Translate provider (cloud, lightweight)
# ============================================================================
FROM base AS provider-google_translate

# Install Google Translate dependencies
COPY services/translation_service/requirements/google_translate.txt /tmp/google_translate.txt
RUN pip install --no-cache-dir -r /tmp/google_translate.txt

# ============================================================================
# Final stage: Select provider based on build argument
# ============================================================================
FROM provider-${PROVIDER} AS final

# Copy project code
COPY src/ /app/src/
COPY services/translation_service/ /app/services/translation_service/

# Expose gRPC port
EXPOSE 50052

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PORT=50052

# Run the server
CMD ["python", "/app/services/translation_service/server.py"]
