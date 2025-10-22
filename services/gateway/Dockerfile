# Gateway Service Dockerfile
# Lightweight - no ML dependencies

FROM python:3.11-slim as base

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project structure
COPY src/generated/ /app/src/generated/
COPY services/gateway/ /app/services/gateway/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/services/gateway/requirements.txt

# Expose HTTP/WebSocket port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PORT=8000

# Run the server
CMD ["python", "/app/services/gateway/server.py"]
