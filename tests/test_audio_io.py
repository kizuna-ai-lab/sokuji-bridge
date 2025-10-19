"""
Unit Tests for Audio I/O System

Tests for microphone, speaker, audio file I/O, and VAD.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from providers.base import AudioChunk, SynthesisResult
from utils.audio_io import list_audio_devices, get_default_input_device, get_default_output_device
from utils.microphone import MicrophoneInput
from utils.speaker import SpeakerOutput
from utils.audio_file import AudioFileReader, AudioFileWriter
from utils.vad import DummyVAD, EnergyVAD, VADResult, SpeechState


# Fixtures

@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / "test_audio.wav"
    yield file_path
    # Cleanup
    if file_path.exists():
        file_path.unlink()
    temp_dir.rmdir()


@pytest.fixture
def sample_audio_chunk():
    """Create a sample audio chunk for testing"""
    sample_rate = 16000
    duration_s = 1.0
    audio_samples = np.random.randn(int(sample_rate * duration_s)).astype(np.float32) * 0.1
    audio_data = (audio_samples * 32768).astype(np.int16).tobytes()

    return AudioChunk(
        data=audio_data,
        sample_rate=sample_rate,
        timestamp=0.0,
        channels=1,
        format="int16",
    )


@pytest.fixture
def sample_synthesis_result():
    """Create a sample synthesis result for testing"""
    sample_rate = 16000
    duration_s = 1.0
    audio_samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, int(sample_rate * duration_s)))
    audio_samples = (audio_samples * 16384).astype(np.int16)
    audio_data = audio_samples.tobytes()

    return SynthesisResult(
        audio_data=audio_data,
        sample_rate=sample_rate,
        text="Test audio",
        voice_id="test",
        timestamp=0.0,
        duration_ms=duration_s * 1000,
    )


# Audio Device Tests

def test_list_audio_devices():
    """Test listing audio devices"""
    devices = list_audio_devices()
    assert isinstance(devices, list)
    # Note: May be empty on CI/CD systems without audio hardware


def test_get_default_input_device():
    """Test getting default input device"""
    device = get_default_input_device()
    # May be None on systems without input devices
    if device is not None:
        assert device.is_input
        assert device.index >= 0


def test_get_default_output_device():
    """Test getting default output device"""
    device = get_default_output_device()
    # May be None on systems without output devices
    if device is not None:
        assert device.is_output
        assert device.index >= 0


# Microphone Tests

@pytest.mark.asyncio
async def test_microphone_initialization():
    """Test microphone initialization"""
    mic = MicrophoneInput(sample_rate=16000, channels=1)
    assert not mic.is_running
    assert mic.sample_rate == 16000
    assert mic.channels == 1


@pytest.mark.asyncio
async def test_microphone_get_devices():
    """Test listing microphone devices"""
    devices = MicrophoneInput.get_devices()
    assert isinstance(devices, list)


# Note: Actual microphone capture tests require audio hardware
# and are skipped in CI/CD environments


# Speaker Tests

@pytest.mark.asyncio
async def test_speaker_initialization():
    """Test speaker initialization"""
    speaker = SpeakerOutput(sample_rate=16000, channels=1)
    assert not speaker.is_running
    assert speaker.sample_rate == 16000
    assert speaker.channels == 1


@pytest.mark.asyncio
async def test_speaker_get_devices():
    """Test listing speaker devices"""
    devices = SpeakerOutput.get_devices()
    assert isinstance(devices, list)


# Note: Actual speaker playback tests require audio hardware
# and are skipped in CI/CD environments


# Audio File Tests

@pytest.mark.asyncio
async def test_audio_file_write_and_read(temp_audio_file, sample_synthesis_result):
    """Test writing and reading audio file"""
    # Write audio file
    writer = AudioFileWriter(str(temp_audio_file), sample_rate=16000, channels=1)
    await writer.start()

    await writer.play(sample_synthesis_result)
    await writer.stop()

    assert temp_audio_file.exists()
    info = writer.get_info()
    assert info['frames_written'] > 0

    # Read audio file back
    reader = AudioFileReader(str(temp_audio_file))
    await reader.start()

    chunks = []
    async for chunk in reader.stream():
        chunks.append(chunk)

    await reader.stop()

    assert len(chunks) > 0
    assert chunks[0].sample_rate == 16000
    assert chunks[0].channels == 1


@pytest.mark.asyncio
async def test_audio_file_reader_metadata(temp_audio_file, sample_synthesis_result):
    """Test audio file reader metadata"""
    # Create test file
    writer = AudioFileWriter(str(temp_audio_file))
    await writer.start()
    await writer.play(sample_synthesis_result)
    await writer.stop()

    # Read metadata
    reader = AudioFileReader(str(temp_audio_file))
    await reader.start()

    info = reader.get_info()
    assert info['sample_rate'] == 16000
    assert info['channels'] == 1
    assert info['duration_ms'] > 0

    await reader.stop()


@pytest.mark.asyncio
async def test_audio_file_writer_multiple_chunks(temp_audio_file, sample_synthesis_result):
    """Test writing multiple audio chunks"""
    writer = AudioFileWriter(str(temp_audio_file))
    await writer.start()

    # Write multiple chunks
    for _ in range(3):
        await writer.play(sample_synthesis_result)

    await writer.stop()

    # Verify file size increased
    assert temp_audio_file.stat().st_size > len(sample_synthesis_result.audio_data)


# VAD Tests

@pytest.mark.asyncio
async def test_dummy_vad(sample_audio_chunk):
    """Test DummyVAD always detects speech"""
    vad = DummyVAD()
    await vad.initialize()

    result = await vad.detect_speech(sample_audio_chunk)

    assert isinstance(result, VADResult)
    assert result.is_speech is True
    assert result.confidence == 1.0
    assert result.state == SpeechState.SPEECH

    await vad.cleanup()


@pytest.mark.asyncio
async def test_energy_vad_loud_audio():
    """Test EnergyVAD with loud audio (should detect speech)"""
    vad = EnergyVAD(energy_threshold=0.05)
    await vad.initialize()

    # Create loud audio
    loud_audio = np.random.randn(16000).astype(np.float32) * 0.5
    loud_data = (loud_audio * 32768).astype(np.int16).tobytes()
    loud_chunk = AudioChunk(loud_data, 16000, 0.0, 1, "int16")

    result = await vad.detect_speech(loud_chunk)

    assert result.is_speech is True
    assert result.confidence > 0.5
    assert result.state == SpeechState.SPEECH

    await vad.cleanup()


@pytest.mark.asyncio
async def test_energy_vad_quiet_audio():
    """Test EnergyVAD with quiet audio (should detect silence)"""
    vad = EnergyVAD(energy_threshold=0.05)
    await vad.initialize()

    # Create quiet audio
    quiet_audio = np.random.randn(16000).astype(np.float32) * 0.01
    quiet_data = (quiet_audio * 32768).astype(np.int16).tobytes()
    quiet_chunk = AudioChunk(quiet_data, 16000, 0.0, 1, "int16")

    result = await vad.detect_speech(quiet_chunk)

    assert result.is_speech is False
    assert result.state == SpeechState.SILENCE

    await vad.cleanup()


@pytest.mark.asyncio
async def test_vad_segmentation():
    """Test VAD audio segmentation"""
    vad = EnergyVAD(energy_threshold=0.05)
    await vad.initialize()

    # Create alternating loud and quiet chunks
    async def audio_stream():
        for i in range(10):
            if i % 2 == 0:
                # Loud chunk (speech)
                audio = np.random.randn(16000).astype(np.float32) * 0.5
            else:
                # Quiet chunk (silence)
                audio = np.random.randn(16000).astype(np.float32) * 0.01

            audio_data = (audio * 32768).astype(np.int16).tobytes()
            yield AudioChunk(audio_data, 16000, i * 1.0, 1, "int16")

    # Segment audio
    segments = []
    async for segment in vad.segment_audio(
        audio_stream(),
        min_speech_duration_ms=100,
        min_silence_duration_ms=500,
    ):
        segments.append(segment)

    # Should have segmented the audio
    assert len(segments) > 0

    await vad.cleanup()


# Integration Tests

@pytest.mark.asyncio
async def test_audio_chunk_properties(sample_audio_chunk):
    """Test AudioChunk properties"""
    assert sample_audio_chunk.sample_rate == 16000
    assert sample_audio_chunk.channels == 1
    assert sample_audio_chunk.format == "int16"
    assert sample_audio_chunk.duration_ms > 0


@pytest.mark.asyncio
async def test_synthesis_result_properties(sample_synthesis_result):
    """Test SynthesisResult properties"""
    assert sample_synthesis_result.sample_rate == 16000
    assert sample_synthesis_result.duration_ms > 0
    assert len(sample_synthesis_result.audio_data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
