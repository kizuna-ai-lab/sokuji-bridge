"""
Unit tests for CosyVoiceProvider
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from providers.base import ProviderStatus, SynthesisResult
from providers.tts.cosyvoice_provider import CosyVoiceProvider


@pytest.fixture
def mock_config():
    """Default test configuration"""
    return {
        "model": "CosyVoice2-0.5B",
        "model_dir": "pretrained_models/CosyVoice2-0.5B",
        "device": "cpu",  # Use CPU for tests
        "inference_mode": "cross_lingual",
        "auto_download": False,  # Don't auto-download in tests
        "streaming": {
            "enabled": True,
        },
        "cross_lingual": {
            "auto_extract_prompt": True,
            "prompt_duration_sec": 3.0,
        },
        "speed": 1.0,
    }


@pytest.fixture
def provider(mock_config):
    """Create CosyVoiceProvider instance"""
    return CosyVoiceProvider(mock_config)


@pytest.fixture
def mock_cosyvoice_model():
    """Mock CosyVoice model"""
    mock_model = Mock()
    mock_model.sample_rate = 22050

    # Mock inference methods to return audio chunks
    def mock_inference(*args, **kwargs):
        # Return mock audio tensor
        import torch
        audio_tensor = torch.randn(1, 22050)  # 1 second of audio
        yield {"tts_speech": audio_tensor}

    mock_model.inference_cross_lingual = Mock(side_effect=mock_inference)
    mock_model.inference_zero_shot = Mock(side_effect=mock_inference)
    mock_model.inference_sft = Mock(side_effect=mock_inference)
    mock_model.inference_instruct = Mock(side_effect=mock_inference)
    mock_model.list_avaliable_spks = Mock(return_value=["speaker1", "speaker2"])

    return mock_model


class TestCosyVoiceProviderInit:
    """Test provider initialization"""

    def test_init_default_config(self, provider):
        """Test initialization with default config"""
        assert provider.model_name == "CosyVoice2-0.5B"
        assert provider.inference_mode == "cross_lingual"
        assert provider.streaming_enabled is True
        assert provider.status == ProviderStatus.UNINITIALIZED

    def test_init_custom_config(self):
        """Test initialization with custom config"""
        config = {
            "model": "CosyVoice-300M-SFT",
            "inference_mode": "sft",
            "device": "cuda",
            "speed": 1.5,
        }
        provider = CosyVoiceProvider(config)
        assert provider.model_name == "CosyVoice-300M-SFT"
        assert provider.inference_mode == "sft"
        assert provider.device == "cuda"
        assert provider.speed == 1.5


class TestCosyVoiceProviderInitialize:
    """Test model loading and initialization"""

    @pytest.mark.asyncio
    async def test_initialize_success(self, provider, mock_cosyvoice_model, tmp_path):
        """Test successful model initialization"""
        # Create mock model directory with config
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")

        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        assert provider.status == ProviderStatus.READY
        assert provider.cosyvoice == mock_cosyvoice_model
        assert provider.sample_rate == 22050
        assert provider.model_version == "2.0"

    @pytest.mark.asyncio
    async def test_initialize_v1_model(self, provider, mock_cosyvoice_model, tmp_path):
        """Test initialization with v1.0 model"""
        # Create mock model directory for v1.0
        model_dir = tmp_path / "CosyVoice-300M"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")

        provider.model_name = "CosyVoice-300M"
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice", return_value=mock_cosyvoice_model):
            await provider.initialize()

        assert provider.status == ProviderStatus.READY
        assert provider.model_version == "1.0"

    @pytest.mark.asyncio
    async def test_initialize_model_not_found(self, provider):
        """Test initialization fails when model not found"""
        provider.model_dir = Path("nonexistent_model")
        provider.auto_download = False

        with pytest.raises(RuntimeError, match="CosyVoice model not found"):
            await provider.initialize()

        assert provider.status == ProviderStatus.UNINITIALIZED

    @pytest.mark.asyncio
    async def test_initialize_already_ready(self, provider, mock_cosyvoice_model, tmp_path):
        """Test initialization skips if already ready"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()
            # Call again
            await provider.initialize()

        assert provider.status == ProviderStatus.READY


class TestCosyVoiceProviderSynthesize:
    """Test synthesis methods"""

    @pytest.mark.asyncio
    async def test_synthesize_cross_lingual_mode(self, provider, mock_cosyvoice_model, tmp_path):
        """Test cross-lingual synthesis (core translation feature)"""
        # Setup
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        # Create mock prompt audio
        prompt_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds at 16kHz

        # Synthesize
        result = await provider.synthesize(
            text="Hello world",
            voice_id="cross_lingual",
            inference_mode="cross_lingual",
            prompt_audio=prompt_audio,
        )

        # Verify
        assert isinstance(result, SynthesisResult)
        assert result.sample_rate == 22050
        assert result.text == "Hello world"
        assert len(result.audio_data) > 0
        assert result.duration_ms > 0
        mock_cosyvoice_model.inference_cross_lingual.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_zero_shot_mode(self, provider, mock_cosyvoice_model, tmp_path):
        """Test zero-shot voice cloning"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        prompt_audio = np.random.randn(16000 * 3).astype(np.float32)
        prompt_text = "This is a reference text"

        result = await provider.synthesize(
            text="Test synthesis",
            voice_id="zero_shot",
            inference_mode="zero_shot",
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )

        assert isinstance(result, SynthesisResult)
        mock_cosyvoice_model.inference_zero_shot.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_sft_mode(self, provider, mock_cosyvoice_model, tmp_path):
        """Test SFT synthesis with preset voices"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        result = await provider.synthesize(
            text="Test SFT",
            voice_id="speaker1",
            inference_mode="sft",
        )

        assert isinstance(result, SynthesisResult)
        mock_cosyvoice_model.inference_sft.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, provider, mock_cosyvoice_model, tmp_path):
        """Test synthesis with empty text returns silence"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        result = await provider.synthesize(
            text="",
            voice_id="default",
        )

        assert isinstance(result, SynthesisResult)
        assert result.duration_ms == 100  # Silence duration
        # Should not call model
        mock_cosyvoice_model.inference_cross_lingual.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_not_initialized(self, provider):
        """Test synthesis fails when not initialized"""
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.synthesize("Test", "voice")


class TestCosyVoiceProviderHelpers:
    """Test helper methods"""

    def test_load_prompt_audio_from_numpy(self, provider):
        """Test loading prompt audio from numpy array"""
        audio = np.random.randn(16000).astype(np.float32)
        result = provider._load_prompt_audio(audio, target_sr=16000)

        assert isinstance(result, np.ndarray)
        assert len(result) == 16000

    def test_load_prompt_audio_none(self, provider):
        """Test loading None prompt audio"""
        result = provider._load_prompt_audio(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_voices(self, provider, mock_cosyvoice_model, tmp_path):
        """Test getting available voices"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        voices = await provider.get_voices()

        assert len(voices) > 0
        # Should include SFT speakers
        sft_voices = [v for v in voices if v["mode"] == "sft"]
        assert len(sft_voices) == 2
        # Should include mode descriptions
        mode_voices = [v for v in voices if v["id"] in ["cross_lingual", "zero_shot", "instruct"]]
        assert len(mode_voices) == 3

    def test_supports_streaming(self, provider):
        """Test streaming support"""
        assert provider.supports_streaming() is True

    def test_supports_voice_cloning(self, provider):
        """Test voice cloning support"""
        assert provider.supports_voice_cloning() is True

    def test_supported_languages(self, provider):
        """Test supported languages"""
        langs = provider.supported_languages()
        assert "zh" in langs
        assert "en" in langs
        assert "ja" in langs
        assert "ko" in langs

    @pytest.mark.asyncio
    async def test_health_check(self, provider, mock_cosyvoice_model, tmp_path):
        """Test health check"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        # Healthy
        is_healthy = await provider.health_check()
        assert is_healthy is True

        # Cleanup makes it unhealthy
        await provider.cleanup()
        is_healthy = await provider.health_check()
        assert is_healthy is False


class TestCosyVoiceProviderCleanup:
    """Test cleanup"""

    @pytest.mark.asyncio
    async def test_cleanup(self, provider, mock_cosyvoice_model, tmp_path):
        """Test cleanup releases resources"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        assert provider.cosyvoice is not None
        assert provider.status == ProviderStatus.READY

        await provider.cleanup()

        assert provider.cosyvoice is None
        assert provider.status == ProviderStatus.STOPPED


class TestCosyVoiceProviderStreaming:
    """Test streaming synthesis"""

    @pytest.mark.asyncio
    async def test_synthesize_stream(self, provider, mock_cosyvoice_model, tmp_path):
        """Test streaming synthesis"""
        model_dir = tmp_path / "CosyVoice2-0.5B"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice.yaml").write_text("# mock config")
        provider.model_dir = model_dir

        with patch("providers.tts.cosyvoice_provider.CosyVoice2", return_value=mock_cosyvoice_model):
            await provider.initialize()

        # Create async text stream
        async def text_stream():
            yield "First chunk"
            yield "Second chunk"
            yield "Third chunk"

        prompt_audio = np.random.randn(16000 * 3).astype(np.float32)

        # Collect results
        results = []
        async for result in provider.synthesize_stream(
            text_stream(),
            voice_id="cross_lingual",
            prompt_audio=prompt_audio,
        ):
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, SynthesisResult)
            assert len(result.audio_data) > 0
