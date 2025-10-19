"""
Speaker Output Implementation

Real-time speaker audio playback using sounddevice.
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any, List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import SynthesisResult
from utils.audio_io import AudioOutput, AudioDevice, list_audio_devices


class SpeakerOutput(AudioOutput):
    """
    Real-time speaker audio output using sounddevice

    Features:
    - Real-time audio playback with configurable buffer size
    - Automatic device selection or manual device specification
    - Queue-based playback for smooth streaming
    - Automatic sample rate conversion if needed
    - Proper start/stop lifecycle management
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_size: int = 2048,
        **kwargs
    ):
        """
        Initialize speaker output

        Args:
            device: Device index (None for default)
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            buffer_size: Output buffer size in samples (default: 2048)
            **kwargs: Additional sounddevice parameters
        """
        super().__init__(kwargs)

        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size

        # Internal state
        self._stream = None
        self._play_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=10)
        self._is_playing = False

    async def start(self) -> None:
        """
        Start speaker output stream

        Raises:
            RuntimeError: If speaker initialization fails
        """
        if self._is_running:
            return

        import sounddevice as sd

        try:
            # Reset state
            self._play_queue = asyncio.Queue(maxsize=10)
            self._is_playing = False

            # Create output stream
            self._stream = sd.OutputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype='int16',
                callback=self._audio_callback,
            )

            self._stream.start()
            self._is_running = True

        except Exception as e:
            raise RuntimeError(f"Failed to start speaker: {e}") from e

    async def stop(self) -> None:
        """Stop speaker output and flush buffers"""
        if not self._is_running:
            return

        # Signal end of playback
        await self._play_queue.put(None)

        # Wait a bit for buffer to drain
        await asyncio.sleep(0.5)

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_running = False
        self._is_playing = False

    async def play(self, audio: SynthesisResult) -> None:
        """
        Play a single audio chunk

        Args:
            audio: Synthesized audio result to play

        Raises:
            RuntimeError: If playback fails
        """
        if not self._is_running:
            raise RuntimeError("Speaker not started. Call start() first.")

        # Convert and resample if needed
        audio_data = await self._prepare_audio(audio)

        # Add to play queue
        await self._play_queue.put(audio_data)

        # Wait for playback to complete
        duration_s = len(audio_data) / (self.sample_rate * self.channels * 2)  # 2 bytes per int16
        await asyncio.sleep(duration_s)

    async def play_stream(self, audio_stream: AsyncIterator[SynthesisResult]) -> None:
        """
        Play audio stream continuously

        Args:
            audio_stream: Async iterator of synthesis results

        Example:
            >>> async for result in pipeline.process_audio_stream(input_stream):
            >>>     await speaker.play_stream([result])
        """
        if not self._is_running:
            raise RuntimeError("Speaker not started. Call start() first.")

        async for audio in audio_stream:
            # Convert and resample if needed
            audio_data = await self._prepare_audio(audio)

            # Add to play queue
            await self._play_queue.put(audio_data)

    async def _prepare_audio(self, audio: SynthesisResult) -> bytes:
        """
        Prepare audio data for playback (resample if needed)

        Args:
            audio: Synthesis result

        Returns:
            Audio data as bytes (int16 format)
        """
        # Check if resampling is needed
        if audio.sample_rate != self.sample_rate:
            # Simple resampling using numpy
            audio_array = np.frombuffer(audio.audio_data, dtype=np.int16)

            # Calculate resampling ratio
            ratio = self.sample_rate / audio.sample_rate
            new_length = int(len(audio_array) * ratio)

            # Linear interpolation resampling
            resampled = np.interp(
                np.linspace(0, len(audio_array), new_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)

            return resampled.tobytes()
        else:
            return audio.audio_data

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Callback function called by sounddevice for each audio block

        Args:
            outdata: Output audio buffer to fill
            frames: Number of frames to generate
            time_info: Timing information
            status: Status flags
        """
        if status:
            print(f"Speaker status: {status}")

        try:
            # Try to get audio data from queue (non-blocking)
            audio_data = self._play_queue.get_nowait()

            if audio_data is None:
                # End of stream signal
                outdata.fill(0)
                self._is_playing = False
                return

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Fill output buffer
            if len(audio_array) >= len(outdata):
                # We have enough data
                outdata[:] = audio_array[:len(outdata)].reshape(-1, self.channels)

                # Put remaining data back in queue if any
                remaining = audio_array[len(outdata):]
                if len(remaining) > 0:
                    self._play_queue.put_nowait(remaining.tobytes())

                self._is_playing = True
            else:
                # Not enough data, pad with zeros
                outdata[:len(audio_array)] = audio_array.reshape(-1, self.channels)
                outdata[len(audio_array):].fill(0)
                self._is_playing = len(audio_array) > 0

        except asyncio.QueueEmpty:
            # No data available, output silence
            outdata.fill(0)
            self._is_playing = False

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """
        List available speaker output devices

        Returns:
            List of available output devices
        """
        return list_audio_devices(output_only=True)

    def get_info(self) -> Dict[str, Any]:
        """
        Get speaker configuration info

        Returns:
            Dictionary with speaker settings
        """
        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_size": self.buffer_size,
            "is_running": self._is_running,
            "is_playing": self._is_playing,
        }

    def get_device_info(self) -> Optional[AudioDevice]:
        """
        Get information about the currently active output device

        Returns:
            AudioDevice object with device details, or None if not running
        """
        if not self._stream:
            return None

        import sounddevice as sd

        # Get the actual device index being used
        device_idx = self._stream.device

        # Query device information directly from sounddevice
        try:
            device_info = sd.query_devices(device_idx)
            return AudioDevice(
                index=device_idx,
                name=device_info['name'],
                channels=device_info['max_output_channels'],
                sample_rate=int(device_info['default_samplerate']),
                is_input=False,
                is_output=True,
                is_default=(device_idx == sd.default.device[1])
            )
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"SpeakerOutput(device={self.device}, "
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"running={self._is_running})"
        )


async def test_speaker():
    """Test speaker playback"""
    print("Testing speaker output...")
    print()

    # List available devices
    devices = SpeakerOutput.get_devices()
    print(f"Available output devices ({len(devices)}):")
    for device in devices:
        print(f"  [{device.index}] {device}")
    print()

    # Create speaker output
    speaker = SpeakerOutput()
    print(f"Created: {speaker}")
    print()

    # Start playback
    print("Starting speaker...")
    await speaker.start()

    # Generate test audio: 1 second sine wave at 440 Hz
    print("Playing test tone (440 Hz, 1 second)...")
    sample_rate = 16000
    duration_s = 1.0
    frequency = 440.0

    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    audio_samples = np.sin(2 * np.pi * frequency * t)
    audio_samples = (audio_samples * 16384).astype(np.int16)  # Scale to int16 range
    audio_data = audio_samples.tobytes()

    # Create SynthesisResult
    from providers.base import SynthesisResult

    test_audio = SynthesisResult(
        audio_data=audio_data,
        sample_rate=sample_rate,
        text="Test tone",
        voice_id="test",
        timestamp=time.time(),
        duration_ms=duration_s * 1000,
    )

    # Play audio
    await speaker.play(test_audio)
    print("Playback complete")
    print()

    # Stop speaker
    await speaker.stop()
    print("Speaker stopped")


if __name__ == "__main__":
    asyncio.run(test_speaker())
