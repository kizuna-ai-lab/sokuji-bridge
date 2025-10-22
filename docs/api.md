# API Reference

Complete API reference for Sokuji-Bridge microservices architecture.

## Overview

Sokuji-Bridge provides two API interfaces:
1. **REST/WebSocket API** (Port 8000) - For client applications
2. **gRPC Services** (Ports 50051-50053) - For internal microservice communication

## Gateway Service API (Port 8000)

The Gateway service orchestrates all microservices and provides a unified API for clients.

### REST Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy" | "degraded",
  "services": {
    "stt": "healthy" | "error",
    "translation": "healthy" | "error",
    "tts": "healthy" | "error"
  }
}
```

#### Translate Text
```http
POST /translate/text
```

**Request Body:**
```json
{
  "text": "Hello world",
  "source_language": "en",
  "target_language": "zh",
  "voice_id": "default"
}
```

**Response:**
```json
{
  "transcription": "Hello world",
  "transcription_language": "en",
  "translation": "你好世界",
  "translation_language": "zh",
  "audio_duration_ms": 1500,
  "total_latency_ms": 250
}
```

#### Get Supported Languages

##### STT Languages
```http
GET /services/stt/languages
```

**Response:**
```json
{
  "languages": ["en", "zh", "ja", "ko", "es", "fr", "de", "ru", ...]
}
```

##### Translation Languages
```http
GET /services/translation/languages
```

**Response:**
```json
{
  "source_languages": ["en", "zh", "ja", ...],
  "target_languages": ["en", "zh", "ja", ...]
}
```

##### TTS Voices
```http
GET /services/tts/voices
```

**Response:**
```json
{
  "voices": [
    {"id": "en_US-lessac-medium", "language": "en", "gender": "male"},
    {"id": "zh_CN-huayan-medium", "language": "zh", "gender": "female"}
  ]
}
```

### WebSocket API

#### Real-time Translation
```
WS /ws/translate
```

**Connection Flow:**

1. **Connect**: Establish WebSocket connection
2. **Configure**: Send configuration message
3. **Stream Audio**: Send audio chunks
4. **Receive Results**: Get transcription, translation, and synthesized audio

**Configuration Message:**
```json
{
  "type": "config",
  "source_language": "zh",
  "target_language": "en",
  "voice_id": "default",
  "sample_rate": 16000,
  "channels": 1
}
```

**Audio Message:**
```json
{
  "type": "audio",
  "data": "base64_encoded_audio_chunk",
  "timestamp": 1234567890
}
```

**Response Messages:**

**Transcription:**
```json
{
  "type": "transcription",
  "text": "你好世界",
  "language": "zh",
  "is_final": true,
  "confidence": 0.95,
  "timestamp": 1234567890
}
```

**Translation:**
```json
{
  "type": "translation",
  "text": "Hello world",
  "source_language": "zh",
  "target_language": "en",
  "timestamp": 1234567891
}
```

**Synthesized Audio:**
```json
{
  "type": "audio",
  "data": "base64_encoded_audio",
  "format": "wav",
  "sample_rate": 22050,
  "timestamp": 1234567892
}
```

## gRPC Services (Internal)

### STT Service (Port 50051)

```protobuf
service STTService {
  // Transcribe single audio chunk
  rpc Transcribe(TranscribeRequest) returns (TranscriptionResult);

  // Stream transcription
  rpc TranscribeStream(stream AudioChunk) returns (stream TranscriptionResult);

  // Get supported languages
  rpc GetSupportedLanguages(Empty) returns (LanguageListResponse);

  // Health check
  rpc HealthCheck(Empty) returns (HealthCheckResponse);

  // Get metrics
  rpc GetMetrics(Empty) returns (MetricsResponse);
}
```

#### Message Types

**TranscribeRequest:**
```protobuf
message TranscribeRequest {
  AudioChunk audio = 1;
  optional string language = 2;  // Language hint
}
```

**AudioChunk:**
```protobuf
message AudioChunk {
  bytes data = 1;               // Raw audio data
  int32 sample_rate = 2;        // Sample rate (e.g., 16000)
  int32 channels = 3;           // Number of channels (1 for mono)
  string format = 4;            // Format (e.g., "int16", "float32")
}
```

**TranscriptionResult:**
```protobuf
message TranscriptionResult {
  string text = 1;              // Transcribed text
  string language = 2;          // Detected language
  float confidence = 3;         // Confidence score (0-1)
  bool is_final = 4;           // Final vs intermediate result
  int64 timestamp_ms = 5;      // Timestamp in milliseconds
}
```

### Translation Service (Port 50052)

```protobuf
service TranslationService {
  // Translate single text
  rpc Translate(TranslateRequest) returns (TranslationResult);

  // Batch translation
  rpc TranslateBatch(TranslateBatchRequest) returns (TranslateBatchResponse);

  // Get supported language pairs
  rpc GetLanguagePairs(Empty) returns (LanguagePairsResponse);

  // Health check
  rpc HealthCheck(Empty) returns (HealthCheckResponse);

  // Get metrics
  rpc GetMetrics(Empty) returns (MetricsResponse);
}
```

#### Message Types

**TranslateRequest:**
```protobuf
message TranslateRequest {
  string text = 1;              // Text to translate
  string source_language = 2;   // Source language code
  string target_language = 3;   // Target language code
  optional string context = 4;  // Additional context
}
```

**TranslationResult:**
```protobuf
message TranslationResult {
  string text = 1;              // Translated text
  string source_language = 2;   // Confirmed source language
  string target_language = 3;   // Target language
  float confidence = 4;         // Translation confidence
  int64 processing_time_ms = 5; // Processing time
}
```

### TTS Service (Port 50053)

```protobuf
service TTSService {
  // Synthesize single text
  rpc Synthesize(SynthesizeRequest) returns (SynthesisResult);

  // Stream synthesis
  rpc SynthesizeStream(stream SynthesizeRequest) returns (stream SynthesisResult);

  // Get available voices
  rpc GetVoices(Empty) returns (VoiceListResponse);

  // Health check
  rpc HealthCheck(Empty) returns (HealthCheckResponse);

  // Get metrics
  rpc GetMetrics(Empty) returns (MetricsResponse);
}
```

#### Message Types

**SynthesizeRequest:**
```protobuf
message SynthesizeRequest {
  string text = 1;              // Text to synthesize
  string language = 2;          // Language code
  optional string voice_id = 3; // Voice ID (provider-specific)
  optional float speed = 4;     // Speech speed (0.5-2.0)
  optional float pitch = 5;     // Voice pitch adjustment
}
```

**SynthesisResult:**
```protobuf
message SynthesisResult {
  bytes audio_data = 1;         // Synthesized audio data
  int32 sample_rate = 2;        // Sample rate (e.g., 22050)
  int32 channels = 3;           // Number of channels
  string format = 4;            // Audio format (e.g., "wav", "mp3")
  int64 duration_ms = 5;        // Audio duration
  int64 processing_time_ms = 6; // Processing time
}
```

## Common Message Types

### HealthCheckResponse
```protobuf
message HealthCheckResponse {
  bool healthy = 1;             // Overall health status
  string status = 2;            // Status description
  string provider = 3;          // Provider name
  int64 uptime_ms = 4;         // Service uptime
  string version = 5;          // Service version
}
```

### MetricsResponse
```protobuf
message MetricsResponse {
  int64 requests_total = 1;     // Total requests processed
  int64 errors_total = 2;       // Total errors
  float avg_latency_ms = 3;     // Average latency
  float p99_latency_ms = 4;     // 99th percentile latency
  int64 uptime_ms = 5;         // Service uptime
}
```

## Error Handling

All API endpoints return standard HTTP status codes:

- **200 OK**: Success
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

Error Response Format:
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

## Rate Limiting

Default rate limits:
- REST API: 100 requests/minute per IP
- WebSocket: 10 concurrent connections per IP
- Audio streaming: 10MB/minute per connection

## Authentication

Currently, the API does not require authentication. For production deployments, implement:
- API key authentication
- JWT tokens for WebSocket connections
- Rate limiting per API key

## Usage Examples

### Python Client Example

```python
import requests
import websocket
import json
import base64

# REST API Example
def translate_text(text, source_lang, target_lang):
    response = requests.post(
        "http://localhost:8000/translate/text",
        json={
            "text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "voice_id": "default"
        }
    )
    return response.json()

# WebSocket Example
def real_time_translate(audio_stream):
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8000/ws/translate")

    # Configure
    ws.send(json.dumps({
        "type": "config",
        "source_language": "zh",
        "target_language": "en"
    }))

    # Stream audio
    for chunk in audio_stream:
        ws.send(json.dumps({
            "type": "audio",
            "data": base64.b64encode(chunk).decode()
        }))

        # Receive results
        result = json.loads(ws.recv())
        if result["type"] == "translation":
            print(f"Translation: {result['text']}")
```

### JavaScript/TypeScript Client Example

```javascript
// REST API Example
async function translateText(text, sourceLang, targetLang) {
  const response = await fetch('http://localhost:8000/translate/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      source_language: sourceLang,
      target_language: targetLang,
      voice_id: 'default'
    })
  });
  return response.json();
}

// WebSocket Example
function realTimeTranslate() {
  const ws = new WebSocket('ws://localhost:8000/ws/translate');

  ws.onopen = () => {
    // Configure
    ws.send(JSON.stringify({
      type: 'config',
      source_language: 'zh',
      target_language: 'en'
    }));
  };

  ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.type === 'translation') {
      console.log('Translation:', result.text);
    }
  };

  return ws;
}
```

## API Versioning

Current version: `v1` (implied in all endpoints)

Future versions will use URL prefixes:
- `/v1/translate/text` (current, default)
- `/v2/translate/text` (future)

## See Also

- [Architecture Documentation](./architecture.md)
- [Provider Guide](./providers.md)
- [Performance Tuning](./performance.md)
- [WebSocket Protocol Details](./MICROSERVICES.md)