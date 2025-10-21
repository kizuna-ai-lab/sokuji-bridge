#!/bin/bash
# Generate Python code from proto files

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Generating Python code from proto files...${NC}"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${PROJECT_ROOT}/proto"
OUTPUT_DIR="${PROJECT_ROOT}/src/generated"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Generate Python code from proto files
python -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUTPUT_DIR}" \
    --grpc_python_out="${OUTPUT_DIR}" \
    --pyi_out="${OUTPUT_DIR}" \
    "${PROTO_DIR}"/*.proto

echo -e "${GREEN}✓ Generated Python code in ${OUTPUT_DIR}${NC}"

# Fix imports in generated files (protobuf imports common.proto without package prefix)
echo -e "${BLUE}Fixing imports in generated files...${NC}"

cd "${OUTPUT_DIR}"

# Replace relative imports with absolute imports for better module resolution
for file in *_pb2.py *_pb2_grpc.py; do
    if [ -f "$file" ]; then
        # Fix: import common_pb2 -> from src.generated import common_pb2
        sed -i 's/^import common_pb2/from src.generated import common_pb2/' "$file"

        # Fix: import stt_pb2 -> from src.generated import stt_pb2
        sed -i 's/^import stt_pb2/from src.generated import stt_pb2/' "$file"

        # Fix: import translation_pb2 -> from src.generated import translation_pb2
        sed -i 's/^import translation_pb2/from src.generated import translation_pb2/' "$file"

        # Fix: import tts_pb2 -> from src.generated import tts_pb2
        sed -i 's/^import tts_pb2/from src.generated import tts_pb2/' "$file"

        echo -e "${GREEN}✓ Fixed imports in $file${NC}"
    fi
done

# Create __init__.py to make it a package
cat > __init__.py << 'EOF'
"""
Generated gRPC code from proto definitions.

This package contains auto-generated code from .proto files.
Do not modify these files directly - regenerate using scripts/generate_protos.sh
"""

from .common_pb2 import (
    AudioChunk,
    TranscriptionResult,
    TranslationResult,
    SynthesisResult,
    HealthCheckResponse,
    MetricsResponse,
    Empty,
)

__all__ = [
    "AudioChunk",
    "TranscriptionResult",
    "TranslationResult",
    "SynthesisResult",
    "HealthCheckResponse",
    "MetricsResponse",
    "Empty",
]
EOF

echo -e "${GREEN}✓ Created __init__.py${NC}"
echo -e "${GREEN}✓ Proto generation complete!${NC}"
