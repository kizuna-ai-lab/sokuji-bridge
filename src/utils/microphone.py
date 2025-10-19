"""
Microphone Input Implementation

Real-time microphone audio capture using sounddevice.
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any, List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import AudioChunk
from utils.audio_io import AudioInput, AudioDevice, list_audio_devices


class MicrophoneInput(AudioInput):
    """
    Real-time microphone input using sounddevice

    Features:
    - Real-time audio capture from system microphones
    - Configurable sample rate, channels, and block size
    - Automatic device selection or manual device specification
    - Queue-based async streaming for smooth processing
    - Proper start/stop lifecycle management
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 1024,
        buffer_size: int = 20,
        **kwargs
    ):
        """
        Initialize microphone input

        Args:
            device: Device index (None for default)
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            block_size: Number of frames per block (default: 1024)
            buffer_size: Maximum number of audio chunks to buffer (default: 20)
            **kwargs: Additional sounddevice parameters
        """
        super().__init__(kwargs)

        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.buffer_size = buffer_size

        # Internal state
        self._stream = None
        self._audio_queue: asyncio.Queue[Optional[AudioChunk]] = None
        self._loop = None

    async def start(self) -> None:
        """
        Start microphone input stream

        Raises:
            RuntimeError: If microphone initialization fails
        """
        if self._is_running:
            return

        import sounddevice as sd

        try:
            # Get current event loop
            self._loop = asyncio.get_running_loop()

            # Create audio queue
            self._audio_queue = asyncio.Queue(maxsize=self.buffer_size)

            # Create input stream
            self._stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype='int16',
                callback=self._audio_callback,
            )

            self._stream.start()
            self._is_running = True

        except Exception as e:
            raise RuntimeError(f"Failed to start microphone: {e}") from e

    async def stop(self) -> None:
        """Stop microphone input stream"""
        if not self._is_running:
            return

        # Signal end of stream
        if self._audio_queue:
            await self._audio_queue.put(None)

        # Wait a bit for queue to drain
        await asyncio.sleep(0.1)

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_running = False
        self._audio_queue = None
        self._loop = None

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """
        Generate audio chunks from microphone input

        Yields:
            AudioChunk objects with audio data

        Raises:
            RuntimeError: If microphone is not started

        Example:
            >>> async for chunk in microphone.stream():
            >>>     print(f"Received {len(chunk.data)} bytes")
        """
        if not self._is_running:
            raise RuntimeError("Microphone not started. Call start() first.")

        while self._is_running:
            # Get audio chunk from queue
            chunk = await self._audio_queue.get()

            # Check for end of stream signal
            if chunk is None:
                break

            yield chunk

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Callback function called by sounddevice for each audio block

        Args:
            indata: Input audio buffer
            frames: Number of frames captured
            time_info: Timing information
            status: Status flags
        """
        if status:
            print(f"Microphone status: {status}", file=sys.stderr)

        if not self._is_running:
            return

        try:
            # Convert numpy array to bytes
            audio_data = indata.copy().tobytes()

            # Create AudioChunk
            chunk = AudioChunk(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=time.time(),
                channels=self.channels,
                format="int16"
            )

            # Add to queue (non-blocking)
            if self._loop and self._audio_queue:
                # Use asyncio to put item in queue from callback thread
                asyncio.run_coroutine_threadsafe(
                    self._audio_queue.put(chunk),
                    self._loop
                )

        except Exception as e:
            print(f"Error in microphone callback: {e}", file=sys.stderr)

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """
        List available microphone input devices

        Returns:
            List of available input devices
        """
        return list_audio_devices(input_only=True)

    def get_info(self) -> Dict[str, Any]:
        """
        Get microphone configuration info

        Returns:
            Dictionary with microphone settings
        """
        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "block_size": self.block_size,
            "buffer_size": self.buffer_size,
            "is_running": self._is_running,
        }

    def get_device_info(self) -> Optional[AudioDevice]:
        """
        Get information about the currently active input device

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
                channels=device_info['max_input_channels'],
                sample_rate=int(device_info['default_samplerate']),
                is_input=True,
                is_output=False,
                is_default=(device_idx == sd.default.device[0])
            )
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"MicrophoneInput(device={self.device}, "
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"running={self._is_running})"
        )


async def test_microphone():
    """Test microphone input"""
    print("Testing microphone input...")
    print()

    # List available devices
    devices = MicrophoneInput.get_devices()
    print(f"Available input devices ({len(devices)}):")
    for device in devices:
        print(f"  [{device.index}] {device}")
    print()

    # Create microphone input
    mic = MicrophoneInput()
    print(f"Created: {mic}")
    print()

    # Start capture
    print("Starting microphone capture...")
    await mic.start()

    device_info = mic.get_device_info()
    if device_info:
        print(f"Active device: {device_info}")
    print()

    # Capture for 5 seconds
    print("Recording for 5 seconds...")
    print("Speak into your microphone!")
    print()

    chunk_count = 0
    total_bytes = 0
    start_time = time.time()
    timeout = 5.0

    try:
        async for chunk in mic.stream():
            chunk_count += 1
            total_bytes += len(chunk.data)

            # Calculate audio energy (simple RMS)
            audio_array = np.frombuffer(chunk.data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            energy = rms / 32768.0  # Normalize to 0-1

            print(f"Chunk {chunk_count}: {len(chunk.data)} bytes, energy: {energy:.4f}")

            # Stop after timeout
            if time.time() - start_time > timeout:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Stop microphone
    await mic.stop()
    print()
    print(f"Capture complete!")
    print(f"  Total chunks: {chunk_count}")
    print(f"  Total bytes: {total_bytes}")
    print(f"  Duration: {timeout:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(test_microphone())
