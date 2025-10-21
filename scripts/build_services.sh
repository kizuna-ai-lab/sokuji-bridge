#!/bin/bash
# Build all Docker services with provider support

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Sokuji-Bridge microservices...${NC}"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

# Check if .env file exists, if not use .env.example
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo -e "${YELLOW}⚠️  No .env file found, using .env.example as template${NC}"
        echo -e "${YELLOW}   Copy .env.example to .env and customize for your deployment${NC}"
    fi
fi

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}✓ Loaded configuration from .env${NC}"
fi

# Get providers from environment or use defaults
STT_PROVIDER=${STT_PROVIDER:-faster_whisper}
TRANSLATION_PROVIDER=${TRANSLATION_PROVIDER:-nllb_local}
TTS_PROVIDER=${TTS_PROVIDER:-piper}

echo -e "${BLUE}Current provider configuration:${NC}"
echo -e "  STT Provider:         ${GREEN}${STT_PROVIDER}${NC}"
echo -e "  Translation Provider: ${GREEN}${TRANSLATION_PROVIDER}${NC}"
echo -e "  TTS Provider:         ${GREEN}${TTS_PROVIDER}${NC}"
echo ""

# Generate proto files first
echo -e "${BLUE}Step 1: Generating gRPC proto files...${NC}"
./scripts/generate_protos.sh

# Build all services
echo -e "${BLUE}Step 2: Building Docker images...${NC}"

services=("stt-service" "translation-service" "tts-service" "gateway")

for service in "${services[@]}"; do
    echo -e "${GREEN}Building ${service}...${NC}"
    docker compose build "${service}" || {
        echo -e "${RED}Failed to build ${service}${NC}"
        exit 1
    }
done

echo -e "${GREEN}✓ All services built successfully!${NC}"
echo ""
echo -e "${BLUE}To start the services:${NC}"
echo -e "  docker compose up -d"
echo ""
echo -e "${BLUE}To view logs:${NC}"
echo -e "  docker compose logs -f"
echo ""
echo -e "${BLUE}To check health:${NC}"
echo -e "  curl http://localhost:8000/health"
echo ""
echo -e "${BLUE}To switch providers:${NC}"
echo -e "  1. Edit .env file and change provider variables"
echo -e "  2. Rebuild: docker compose build [service-name]"
echo -e "  3. Restart: docker compose up -d [service-name]"
echo ""
echo -e "${BLUE}Example - Switch TTS to XTTS:${NC}"
echo -e "  echo 'TTS_PROVIDER=xtts' >> .env"
echo -e "  docker compose build tts-service"
echo -e "  docker compose up -d tts-service"
