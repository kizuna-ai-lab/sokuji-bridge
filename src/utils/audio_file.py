"""
Audio File I/O Implementation

Read and write audio files in various formats using soundfile.
"""

import time
from typing import AsyncIterator, Optional, Dict, Any, List
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import AudioChunk, SynthesisResult
from utils.audio_io import AudioInput, AudioOutput, AudioDevice


class AudioFileReader(AudioInput):
    """
    Read audio files and stream as AudioChunk objects

    Supports: WAV, FLAC, OGG formats via soundfile
    Features:
    - Stream large files in chunks to avoid memory issues
    - Automatic format detection
    - Sample rate and channel configuration
    - Seek support for random access
    """

    def __init__(
        self,
        file_path: str,
        chunk_duration_ms: float = 1000.0,
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None,
    ):
        """
        Initialize audio file reader

        Args:
            file_path: Path to audio file
            chunk_duration_ms: Duration of each chunk in milliseconds (default: 1000ms)
            target_sample_rate: Convert to this sample rate (None to keep original)
            target_channels: Convert to this many channels (None to keep original)
        """
        super().__init__()

        self.file_path = Path(file_path)
        self.chunk_duration_ms = chunk_duration_ms
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

        # File metadata
        self._soundfile = None
        self._original_sample_rate = None
        self._original_channels = None
        self._frames_total = None
        self._current_frame = 0

    async def start(self) -> None:
        """
        Open audio file for reading

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If file cannot be opened
        """
        if self._is_running:
            return

        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        import soundfile as sf

        try:
            self._soundfile = sf.SoundFile(str(self.file_path), mode='r')
            self._original_sample_rate = self._soundfile.samplerate
            self._original_channels = self._soundfile.channels
            self._frames_total = len(self._soundfile)
            self._current_frame = 0

            # Use original values if targets not specified
            if self.target_sample_rate is None:
                self.target_sample_rate = self._original_sample_rate
            if self.target_channels is None:
                self.target_channels = self._original_channels

            self._is_running = True

        except Exception as e:
            raise RuntimeError(f"Failed to open audio file: {e}") from e

    async def stop(self) -> None:
        """Close audio file"""
        if not self._is_running:
            return

        if self._soundfile:
            self._soundfile.close()
            self._soundfile = None

        self._is_running = False

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """
        Stream audio file as chunks

        Yields:
            AudioChunk objects containing file audio

        Example:
            >>> reader = AudioFileReader("audio.wav")
            >>> await reader.start()
            >>> async for chunk in reader.stream():
            >>>     print(f"Read {chunk.duration_ms}ms from file")
            >>> await reader.stop()
        """
        if not self._is_running:
            raise RuntimeError("File not opened. Call start() first.")

        # Calculate chunk size in frames
        chunk_frames = int(self._original_sample_rate * self.chunk_duration_ms / 1000)

        while self._current_frame < self._frames_total:
            # Read chunk from file
            audio_data = self._soundfile.read(chunk_frames, dtype='float32')

            if len(audio_data) == 0:
                break

            # Convert to target sample rate and channels if needed
            audio_data = self._convert_audio(audio_data)

            # Convert to int16
            audio_int16 = (audio_data * 32768).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # Create AudioChunk
            chunk = AudioChunk(
                data=audio_bytes,
                sample_rate=self.target_sample_rate,
                timestamp=self._current_frame / self._original_sample_rate,
                channels=self.target_channels,
                format="int16",
            )

            self._current_frame += len(audio_data)
            yield chunk

    def _convert_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to target sample rate and channels

        Args:
            audio: Audio data as numpy array

        Returns:
            Converted audio data
        """
        # Convert channels if needed
        if self._original_channels != self.target_channels:
            if self._original_channels == 1 and self.target_channels == 2:
                # Mono to stereo
                audio = np.stack([audio, audio], axis=-1)
            elif self._original_channels == 2 and self.target_channels == 1:
                # Stereo to mono
                audio = np.mean(audio, axis=-1)
            elif self._original_channels == 2:
                # Multi-channel, take first channel only
                audio = audio[:, 0]

        # Resample if needed
        if self._original_sample_rate != self.target_sample_rate:
            ratio = self.target_sample_rate / self._original_sample_rate

            # Handle multi-channel audio
            if audio.ndim == 1:
                # Mono audio
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio
                )
            else:
                # Multi-channel audio - resample each channel separately
                new_length = int(audio.shape[0] * ratio)
                resampled_channels = []
                for ch in range(audio.shape[1]):
                    resampled = np.interp(
                        np.linspace(0, audio.shape[0], new_length),
                        np.arange(audio.shape[0]),
                        audio[:, ch]
                    )
                    resampled_channels.append(resampled)
                audio = np.stack(resampled_channels, axis=-1)

        return audio

    def seek(self, position_ms: float) -> None:
        """
        Seek to position in file

        Args:
            position_ms: Position in milliseconds
        """
        if not self._is_running:
            raise RuntimeError("File not opened")

        frame = int(position_ms * self._original_sample_rate / 1000)
        self._soundfile.seek(frame)
        self._current_frame = frame

    def get_duration_ms(self) -> float:
        """Get total duration of audio file in milliseconds"""
        if not self._is_running:
            raise RuntimeError("File not opened")
        return (self._frames_total / self._original_sample_rate) * 1000

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """Not applicable for file reader"""
        return []

    def get_info(self) -> Dict[str, Any]:
        """Get file information"""
        return {
            "file_path": str(self.file_path),
            "sample_rate": self._original_sample_rate,
            "channels": self._original_channels,
            "frames": self._frames_total,
            "duration_ms": self.get_duration_ms() if self._is_running else 0,
            "is_running": self._is_running,
        }

    def __repr__(self) -> str:
        return f"AudioFileReader({self.file_path.name})"


class AudioFileWriter(AudioOutput):
    """
    Write audio to file

    Supports: WAV format via soundfile
    Features:
    - Write single audio chunk or stream
    - Automatic file creation and management
    - Metadata preservation
    """

    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "WAV",
        subtype: str = "PCM_16",
    ):
        """
        Initialize audio file writer

        Args:
            file_path: Output file path
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of channels (default: 1)
            format: Audio format (default: "WAV")
            subtype: Audio subtype (default: "PCM_16")
        """
        super().__init__()

        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.subtype = subtype

        self._soundfile = None
        self._frames_written = 0

    async def start(self) -> None:
        """
        Create audio file for writing

        Raises:
            RuntimeError: If file cannot be created
        """
        if self._is_running:
            return

        import soundfile as sf

        try:
            # Create parent directory if needed
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Open file for writing
            self._soundfile = sf.SoundFile(
                str(self.file_path),
                mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                format=self.format,
                subtype=self.subtype,
            )

            self._frames_written = 0
            self._is_running = True

        except Exception as e:
            raise RuntimeError(f"Failed to create audio file: {e}") from e

    async def stop(self) -> None:
        """Close audio file and finalize"""
        if not self._is_running:
            return

        if self._soundfile:
            self._soundfile.close()
            self._soundfile = None

        self._is_running = False

    async def play(self, audio: SynthesisResult) -> None:
        """
        Write audio to file (named 'play' for interface compatibility)

        Args:
            audio: Synthesized audio result to write

        Raises:
            RuntimeError: If write fails
        """
        if not self._is_running:
            raise RuntimeError("File not opened. Call start() first.")

        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio.audio_data, dtype=np.int16)

            # Resample if needed
            if audio.sample_rate != self.sample_rate:
                ratio = self.sample_rate / audio.sample_rate
                new_length = int(len(audio_array) * ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), new_length),
                    np.arange(len(audio_array)),
                    audio_array
                ).astype(np.int16)

            # Convert to float32 for soundfile
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Write to file
            self._soundfile.write(audio_float)
            self._frames_written += len(audio_float)

        except Exception as e:
            raise RuntimeError(f"Failed to write audio: {e}") from e

    async def play_stream(self, audio_stream: AsyncIterator[SynthesisResult]) -> None:
        """
        Write audio stream to file

        Args:
            audio_stream: Async iterator of synthesis results
        """
        if not self._is_running:
            raise RuntimeError("File not opened. Call start() first.")

        async for audio in audio_stream:
            await self.play(audio)

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """Not applicable for file writer"""
        return []

    def get_info(self) -> Dict[str, Any]:
        """Get file writer information"""
        return {
            "file_path": str(self.file_path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "frames_written": self._frames_written,
            "duration_ms": (self._frames_written / self.sample_rate) * 1000,
            "is_running": self._is_running,
        }

    def __repr__(self) -> str:
        return f"AudioFileWriter({self.file_path.name})"


async def test_audio_file():
    """Test audio file read/write"""
    print("Testing audio file I/O...")
    print()

    # Create temporary test file
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test_audio.wav"

    # Test Writer
    print("Testing AudioFileWriter...")
    writer = AudioFileWriter(str(test_file))
    print(f"Created: {writer}")
    await writer.start()

    # Generate test audio: 2 seconds sine wave at 440 Hz
    sample_rate = 16000
    duration_s = 2.0
    frequency = 440.0

    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    audio_samples = np.sin(2 * np.pi * frequency * t)
    audio_samples = (audio_samples * 16384).astype(np.int16)
    audio_data = audio_samples.tobytes()

    # Write audio
    test_audio = SynthesisResult(
        audio_data=audio_data,
        sample_rate=sample_rate,
        text="Test audio",
        voice_id="test",
        timestamp=time.time(),
        duration_ms=duration_s * 1000,
    )

    await writer.play(test_audio)
    await writer.stop()
    print(f"✓ Wrote {writer.get_info()['frames_written']} frames")
    print()

    # Test Reader
    print("Testing AudioFileReader...")
    reader = AudioFileReader(str(test_file))
    print(f"Created: {reader}")
    await reader.start()
    print(f"File info: {reader.get_info()}")
    print()

    # Read back audio
    chunk_count = 0
    async for chunk in reader.stream():
        chunk_count += 1
        print(f"Chunk {chunk_count}: {chunk.duration_ms:.1f}ms")

    await reader.stop()
    print(f"✓ Read {chunk_count} chunks")
    print()

    # Cleanup
    test_file.unlink()
    temp_dir.rmdir()
    print("Test complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_audio_file())
