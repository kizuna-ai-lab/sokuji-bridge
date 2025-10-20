"""
CosyVoice2 TTS Provider

Multi-lingual TTS with cross-lingual voice synthesis.
Optimized for translation scenarios with native streaming support.

Uses CosyVoice2 from https://github.com/FunAudioLLM/CosyVoice
"""

import time
import io
import numpy as np
from typing import AsyncIterator, Optional, Dict, Any, Union
from pathlib import Path

from providers.base import (
    TTSProvider,
    SynthesisResult,
    ProviderStatus,
)


class CosyVoiceProvider(TTSProvider):
    """
    CosyVoice2 TTS provider with cross-lingual synthesis

    Features:
    - Cross-lingual voice synthesis (for translation scenarios)
    - Native streaming support with <200ms latency
    - Zero-shot voice cloning
    - Multi-lingual support (zh, en, ja, ko + dialects)
    - Multiple inference modes (SFT, Zero-shot, Cross-lingual, Instruct)

    Translation Workflow:
        STT (source audio + text)
          → Translation (target text)
          → TTS (target text + source audio as prompt)
          → Output (target text in source voice)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CosyVoice provider

        Args:
            config: Configuration dictionary with:
                - model: Model name (e.g., "CosyVoice2-0.5B", "CosyVoice-300M")
                - model_dir: Path to model directory
                - inference_mode: Default mode (sft|zero_shot|cross_lingual|instruct)
                - device: Device for inference (cuda|cpu)
                - streaming: Streaming configuration
                - cross_lingual: Cross-lingual mode config
                - zero_shot: Zero-shot mode config
                - sft: SFT mode config
                - instruct: Instruct mode config
                - speed: Speech speed multiplier (default: 1.0)
                - auto_download: Auto-download model if not found
        """
        super().__init__(config)

        # Model configuration
        self.model_name = config.get("model", "CosyVoice2-0.5B")
        self.model_dir = Path(config.get("model_dir", f"pretrained_models/{self.model_name}"))
        self.device = config.get("device", "cuda")
        self.auto_download = config.get("auto_download", True)
        self.download_source = config.get("download_source", "modelscope")

        # Inference mode (default: cross_lingual for translation)
        self.inference_mode = config.get("inference_mode", "cross_lingual")

        # Mode-specific configurations
        self.cross_lingual_config = config.get("cross_lingual", {})
        self.zero_shot_config = config.get("zero_shot", {})
        self.sft_config = config.get("sft", {})
        self.instruct_config = config.get("instruct", {})

        # Streaming configuration
        self.streaming_config = config.get("streaming", {})
        self.streaming_enabled = self.streaming_config.get("enabled", True)
        self.token_hop_len = self.streaming_config.get("token_hop_len", 50)

        # Synthesis parameters
        self.speed = config.get("speed", 1.0)
        self.text_frontend = config.get("text_frontend", True)

        # Model instance (initialized in initialize())
        self.cosyvoice = None
        self.sample_rate = 22050  # Will be updated after model load

        # Supported model mapping
        self._model_mapping = {
            "CosyVoice2-0.5B": "iic/CosyVoice2-0.5B",
            "CosyVoice-300M": "iic/CosyVoice-300M",
            "CosyVoice-300M-SFT": "iic/CosyVoice-300M-SFT",
            "CosyVoice-300M-Instruct": "iic/CosyVoice-300M-Instruct",
        }

        # Language support
        self._supported_languages = [
            "zh", "en", "ja", "ko",  # Main languages
            "yue", "zh-yue",  # Cantonese
            "zh-sichuan", "zh-shanghai", "zh-tianjin", "zh-wuhan",  # Dialects
        ]

    def _download_model(self, model_name: str) -> None:
        """
        Download CosyVoice model using ModelScope or HuggingFace

        Args:
            model_name: Name of the model to download
        """
        try:
            model_id = self._model_mapping.get(model_name, model_name)
            print(f"📥 Downloading CosyVoice model: {model_id}")
            print(f"📁 Download directory: {self.model_dir}")

            self.model_dir.mkdir(parents=True, exist_ok=True)

            if self.download_source == "modelscope":
                from modelscope import snapshot_download
                snapshot_download(model_id, local_dir=str(self.model_dir))
            else:  # huggingface
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_id.replace("iic/", "FunAudioLLM/"),
                    local_dir=str(self.model_dir)
                )

            print(f"✓ Model {model_name} downloaded successfully")

        except ImportError as e:
            raise RuntimeError(
                f"Missing download dependency. Install with: "
                f"pip install {'modelscope' if self.download_source == 'modelscope' else 'huggingface-hub'}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_name}: {e}") from e

    async def initialize(self) -> None:
        """Initialize CosyVoice model"""
        if self.status == ProviderStatus.READY:
            return

        self.status = ProviderStatus.INITIALIZING

        try:
            # Auto-download if model doesn't exist
            if not self.model_dir.exists() or not (self.model_dir / "cosyvoice.yaml").exists():
                if self.auto_download:
                    print(f"⚠️  Model not found at {self.model_dir}")
                    print(f"🔄 Attempting to download {self.model_name}...")
                    self._download_model(self.model_name)
                else:
                    raise FileNotFoundError(
                        f"CosyVoice model not found: {self.model_dir}. "
                        f"Set auto_download=true or download manually."
                    )

            # Load appropriate model class based on model name
            if "CosyVoice2" in self.model_name or "cosyvoice2" in str(self.model_dir).lower():
                print(f"📂 Loading CosyVoice2 model from: {self.model_dir}")
                from cosyvoice.cli.cosyvoice import CosyVoice2
                self.cosyvoice = CosyVoice2(str(self.model_dir))
                self.model_version = "2.0"
            else:
                print(f"📂 Loading CosyVoice v1.0 model from: {self.model_dir}")
                from cosyvoice.cli.cosyvoice import CosyVoice
                self.cosyvoice = CosyVoice(str(self.model_dir))
                self.model_version = "1.0"

            # Update sample rate from loaded model
            self.sample_rate = self.cosyvoice.sample_rate

            self.status = ProviderStatus.READY
            print(f"✓ CosyVoice initialized: {self.model_name} (v{self.model_version}) @ {self.sample_rate}Hz")
            print(f"  Inference mode: {self.inference_mode}")
            print(f"  Streaming: {'enabled' if self.streaming_enabled else 'disabled'}")

        except ImportError as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(
                "cosyvoice not installed. Install with: pip install cosyvoice"
            ) from e
        except Exception as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(f"Failed to initialize CosyVoice: {e}") from e

    async def cleanup(self) -> None:
        """Clean up CosyVoice model"""
        if self.cosyvoice:
            del self.cosyvoice
            self.cosyvoice = None

        self.status = ProviderStatus.STOPPED

    def _load_prompt_audio(
        self,
        prompt_audio: Optional[Union[str, Path, bytes, np.ndarray]],
        target_sr: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Load and preprocess prompt audio for voice cloning/cross-lingual

        Args:
            prompt_audio: Audio data (path, bytes, or numpy array)
            target_sr: Target sample rate (CosyVoice requires 16kHz)

        Returns:
            Processed audio array at 16kHz or None
        """
        if prompt_audio is None:
            return None

        try:
            import torchaudio

            # Handle different input types
            if isinstance(prompt_audio, (str, Path)):
                # Load from file path
                audio, sr = torchaudio.load(str(prompt_audio))
            elif isinstance(prompt_audio, bytes):
                # Load from bytes
                audio_buffer = io.BytesIO(prompt_audio)
                audio, sr = torchaudio.load(audio_buffer)
            elif isinstance(prompt_audio, np.ndarray):
                # Already numpy array, assume 16kHz if not specified
                audio = prompt_audio
                sr = target_sr
            else:
                raise ValueError(f"Unsupported prompt_audio type: {type(prompt_audio)}")

            # Convert to numpy if tensor
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()

            # Convert stereo to mono if needed
            if audio.ndim > 1 and audio.shape[0] > 1:
                audio = audio.mean(axis=0)
            elif audio.ndim > 1:
                audio = audio[0]

            # Resample if needed
            if sr != target_sr:
                import torchaudio.functional as F
                import torch
                audio_tensor = torch.from_numpy(audio).float()
                audio_tensor = F.resample(audio_tensor, sr, target_sr)
                audio = audio_tensor.numpy()

            return audio

        except Exception as e:
            print(f"⚠️ Warning: Failed to load prompt audio: {e}")
            return None

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (mode-dependent)
            language: Optional language hint
            **kwargs: Additional parameters:
                - inference_mode: Override default mode
                - prompt_audio: Reference audio for zero_shot/cross_lingual
                - prompt_text: Reference text for zero_shot
                - instruct_text: Instruction for instruct mode
                - speed: Speech speed multiplier
                - stream: Enable streaming (default: self.streaming_enabled)

        Returns:
            Synthesis result with audio data
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        if not text or not text.strip():
            # Return silence for empty text
            silence = np.zeros(int(self.sample_rate * 0.1), dtype=np.int16)
            return SynthesisResult(
                audio_data=silence.tobytes(),
                sample_rate=self.sample_rate,
                text=text,
                voice_id=voice_id,
                timestamp=time.time(),
                duration_ms=100,
                format="int16",
            )

        start_time = time.time()

        try:
            # Determine inference mode
            mode = kwargs.get("inference_mode", self.inference_mode)
            speed = kwargs.get("speed", self.speed)
            stream = kwargs.get("stream", self.streaming_enabled)

            # Collect all audio chunks
            audio_chunks = []

            if mode == "cross_lingual":
                # Cross-lingual: Use source audio prompt to speak target language
                prompt_audio = kwargs.get("prompt_audio") or self.cross_lingual_config.get("prompt_audio")
                prompt_speech_16k = self._load_prompt_audio(prompt_audio)

                if prompt_speech_16k is None:
                    raise ValueError("cross_lingual mode requires prompt_audio")

                # CosyVoice cross-lingual API
                for chunk in self.cosyvoice.inference_cross_lingual(
                    tts_text=text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=stream,
                    speed=speed,
                    text_frontend=self.text_frontend,
                ):
                    audio_chunks.append(chunk['tts_speech'])

            elif mode == "zero_shot":
                # Zero-shot: Clone voice from audio + text prompt
                prompt_audio = kwargs.get("prompt_audio") or self.zero_shot_config.get("prompt_audio")
                prompt_text = kwargs.get("prompt_text") or self.zero_shot_config.get("prompt_text")
                prompt_speech_16k = self._load_prompt_audio(prompt_audio)

                if prompt_speech_16k is None or not prompt_text:
                    raise ValueError("zero_shot mode requires prompt_audio and prompt_text")

                for chunk in self.cosyvoice.inference_zero_shot(
                    tts_text=text,
                    prompt_text=prompt_text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=stream,
                    speed=speed,
                    text_frontend=self.text_frontend,
                ):
                    audio_chunks.append(chunk['tts_speech'])

            elif mode == "sft":
                # SFT: Standard synthesis with preset voices
                sft_speaker = kwargs.get("speaker", voice_id) or self.sft_config.get("speaker", "default")

                for chunk in self.cosyvoice.inference_sft(
                    tts_text=text,
                    sft_spk=sft_speaker,
                    stream=stream,
                    speed=speed,
                    text_frontend=self.text_frontend,
                ):
                    audio_chunks.append(chunk['tts_speech'])

            elif mode == "instruct":
                # Instruct: Synthesis with emotion/style control
                instruct_text = kwargs.get("instruct_text") or self.instruct_config.get("instruct_text")
                sft_speaker = kwargs.get("speaker") or self.instruct_config.get("sft_speaker", "default")
                prompt_audio = kwargs.get("prompt_audio") or self.instruct_config.get("prompt_audio")
                prompt_speech_16k = self._load_prompt_audio(prompt_audio)

                if not instruct_text or prompt_speech_16k is None:
                    raise ValueError("instruct mode requires instruct_text and prompt_audio")

                for chunk in self.cosyvoice.inference_instruct(
                    tts_text=text,
                    sft_spk=sft_speaker,
                    instruct_text=instruct_text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=stream,
                    speed=speed,
                    text_frontend=self.text_frontend,
                ):
                    audio_chunks.append(chunk['tts_speech'])

            else:
                raise ValueError(f"Unknown inference mode: {mode}")

            # Concatenate audio chunks
            import torch
            if audio_chunks:
                audio_tensor = torch.cat(audio_chunks, dim=1)
                audio_np = audio_tensor.cpu().numpy()

                # Convert to int16
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_data = audio_int16.tobytes()
            else:
                # Empty result
                audio_data = np.zeros(1, dtype=np.int16).tobytes()

            # Calculate metrics
            sample_count = len(audio_data) // 2  # int16 = 2 bytes per sample
            duration_ms = (sample_count / self.sample_rate) * 1000
            synthesis_time_ms = (time.time() - start_time) * 1000

            self._record_request(synthesis_time_ms, error=False)

            return SynthesisResult(
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                text=text,
                voice_id=voice_id or mode,
                timestamp=time.time(),
                duration_ms=duration_ms,
                format="int16",
            )

        except Exception as e:
            self._record_request(0, error=True)
            raise RuntimeError(f"Synthesis failed ({mode} mode): {e}") from e

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        voice_id: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[SynthesisResult]:
        """
        Synthesize speech from text stream

        Args:
            text_stream: Async iterator of texts
            voice_id: Voice identifier
            language: Optional language hint
            **kwargs: Additional parameters (see synthesize())

        Yields:
            Synthesis results as they become available
        """
        # For text streaming, process each chunk separately
        # CosyVoice handles internal streaming within each chunk
        async for text in text_stream:
            if text and text.strip():
                result = await self.synthesize(
                    text, voice_id, language,
                    stream=True,  # Enable internal streaming
                    **kwargs
                )
                yield result

    async def get_voices(self) -> list[Dict[str, Any]]:
        """
        Get available voices

        Returns:
            List of voice dictionaries
        """
        voices = []

        # For SFT mode, list preset speakers
        if hasattr(self.cosyvoice, 'list_avaliable_spks'):
            try:
                spk_list = self.cosyvoice.list_avaliable_spks()
                for spk in spk_list:
                    voices.append({
                        "id": spk,
                        "name": spk,
                        "language": "multi",
                        "mode": "sft",
                        "description": "Preset voice (SFT mode)",
                    })
            except Exception:
                pass

        # Add mode descriptions
        voices.extend([
            {
                "id": "cross_lingual",
                "name": "Cross-lingual (Translation)",
                "language": "multi",
                "mode": "cross_lingual",
                "description": "Speak target language with source voice (requires prompt_audio)",
            },
            {
                "id": "zero_shot",
                "name": "Zero-shot Clone",
                "language": "multi",
                "mode": "zero_shot",
                "description": "Clone voice from audio sample (requires prompt_audio + prompt_text)",
            },
            {
                "id": "instruct",
                "name": "Instruct Mode",
                "language": "multi",
                "mode": "instruct",
                "description": "Synthesis with emotion/style control (requires instruct_text + prompt_audio)",
            },
        ])

        return voices

    def supports_streaming(self) -> bool:
        """CosyVoice supports native streaming synthesis"""
        return True

    def supports_voice_cloning(self) -> bool:
        """CosyVoice supports zero-shot voice cloning"""
        return True

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return self._supported_languages.copy()

    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        if self.status != ProviderStatus.READY:
            return False

        if self.cosyvoice is None:
            return False

        # Test synthesis with SFT mode (simplest)
        try:
            test_text = "Hello"
            for chunk in self.cosyvoice.inference_sft(
                tts_text=test_text,
                sft_spk="default",
                stream=False,
            ):
                # Just check if we get output
                return chunk['tts_speech'].numel() > 0
            return False
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"CosyVoiceProvider(model={self.model_name}, "
            f"version={getattr(self, 'model_version', 'unknown')}, "
            f"mode={self.inference_mode}, "
            f"streaming={self.streaming_enabled}, "
            f"status={self.status.value})"
        )


# Register provider
from providers.base import register_provider
register_provider("tts", "cosyvoice", CosyVoiceProvider)
