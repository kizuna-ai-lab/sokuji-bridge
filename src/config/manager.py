"""
Configuration Manager for Sokuji-Bridge

Handles loading, validation, and management of configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from .schemas import SokujiBridgeConfig, get_profile, PROFILES


class ConfigManager:
    """Configuration manager for loading and managing configs"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._config: Optional[SokujiBridgeConfig] = None

        # Load environment variables
        load_dotenv()

    @classmethod
    def from_profile(cls, profile_name: str) -> "ConfigManager":
        """
        Create ConfigManager from predefined profile

        Args:
            profile_name: Profile name ("fast", "hybrid", "quality", "cpu")

        Returns:
            ConfigManager with profile configuration

        Example:
            >>> manager = ConfigManager.from_profile("fast")
            >>> config = manager.get_config()
        """
        manager = cls()
        manager._config = get_profile(profile_name)
        manager._inject_env_vars()
        return manager

    @classmethod
    def from_file(cls, config_path: Path) -> "ConfigManager":
        """
        Create ConfigManager from configuration file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ConfigManager with loaded configuration

        Example:
            >>> manager = ConfigManager.from_file("configs/custom.yaml")
            >>> config = manager.get_config()
        """
        manager = cls(config_path=config_path)
        manager.load()
        return manager

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigManager":
        """
        Create ConfigManager from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            ConfigManager with parsed configuration
        """
        manager = cls()
        manager._config = SokujiBridgeConfig(**config_dict)
        manager._inject_env_vars()
        return manager

    def load(self) -> SokujiBridgeConfig:
        """
        Load configuration from file

        Returns:
            Loaded and validated configuration

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If configuration is invalid
        """
        if not self.config_path:
            raise ValueError("No configuration path specified")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self._config = SokujiBridgeConfig(**config_dict)
        self._inject_env_vars()

        return self._config

    def _inject_env_vars(self) -> None:
        """Inject environment variables into API keys configuration"""
        if not self._config:
            return

        # Inject API keys from environment
        self._config.api_keys.deepl = os.getenv("DEEPL_API_KEY")
        self._config.api_keys.openai = os.getenv("OPENAI_API_KEY")
        self._config.api_keys.elevenlabs = os.getenv("ELEVENLABS_API_KEY")
        self._config.api_keys.azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
        self._config.api_keys.azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
        self._config.api_keys.google_application_credentials = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

    def get_config(self) -> SokujiBridgeConfig:
        """
        Get current configuration

        Returns:
            Current configuration

        Raises:
            RuntimeError: If configuration not loaded
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config

    def save(self, output_path: Path) -> None:
        """
        Save current configuration to file

        Args:
            output_path: Path to save configuration file

        Raises:
            RuntimeError: If configuration not loaded
        """
        if not self._config:
            raise RuntimeError("No configuration to save")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(self._config.model_dump_yaml())

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not self._config:
            return False, ["Configuration not loaded"]

        errors = []

        # Validate API keys if needed
        if self._config.translation.provider == "deepl_api":
            if not self._config.api_keys.deepl:
                errors.append("DeepL API key not configured (DEEPL_API_KEY)")

        if self._config.translation.provider == "openai_translator":
            if not self._config.api_keys.openai:
                errors.append("OpenAI API key not configured (OPENAI_API_KEY)")

        if self._config.tts.provider == "elevenlabs":
            if not self._config.api_keys.elevenlabs:
                errors.append("ElevenLabs API key not configured (ELEVENLABS_API_KEY)")

        if self._config.stt.provider == "azure_stt":
            if not self._config.api_keys.azure_speech_key:
                errors.append("Azure Speech key not configured (AZURE_SPEECH_KEY)")
            if not self._config.api_keys.azure_speech_region:
                errors.append("Azure Speech region not configured (AZURE_SPEECH_REGION)")

        # Validate device compatibility
        if self._config.stt.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("CUDA device specified but not available for STT")
            except ImportError:
                errors.append("PyTorch not installed, cannot use CUDA for STT")

        return len(errors) == 0, errors

    def list_profiles(self) -> list[str]:
        """List available configuration profiles"""
        return list(PROFILES.keys())

    def get_provider_config(self, category: str) -> Dict[str, Any]:
        """
        Get provider configuration for a specific category

        Args:
            category: Category name ("stt", "translation", "tts")

        Returns:
            Provider configuration dictionary
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        if category == "stt":
            return {
                "provider": self._config.stt.provider,
                "config": self._config.stt.config,
                "device": self._config.stt.device,
            }
        elif category == "translation":
            return {
                "provider": self._config.translation.provider,
                "config": self._config.translation.config,
                "device": self._config.translation.device,
            }
        elif category == "tts":
            return {
                "provider": self._config.tts.provider,
                "config": self._config.tts.config,
                "device": self._config.tts.device,
            }
        else:
            raise ValueError(f"Unknown category: {category}")

    def update_provider(
        self,
        category: str,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update provider configuration at runtime

        Args:
            category: Category name ("stt", "translation", "tts")
            provider: Provider name
            config: Optional provider configuration

        Example:
            >>> manager.update_provider("translation", "deepl_api", {"formality": "more"})
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        if category == "stt":
            self._config.stt.provider = provider
            if config:
                self._config.stt.config.update(config)
        elif category == "translation":
            self._config.translation.provider = provider
            if config:
                self._config.translation.config.update(config)
        elif category == "tts":
            self._config.tts.provider = provider
            if config:
                self._config.tts.config.update(config)
        else:
            raise ValueError(f"Unknown category: {category}")

    def __repr__(self) -> str:
        if self._config:
            return (
                f"ConfigManager(pipeline={self._config.pipeline.name}, "
                f"stt={self._config.stt.provider}, "
                f"translation={self._config.translation.provider}, "
                f"tts={self._config.tts.provider})"
            )
        return "ConfigManager(not loaded)"


# Convenience functions

def load_config(
    config_path: Optional[Path] = None,
    profile: Optional[str] = None,
) -> SokujiBridgeConfig:
    """
    Load configuration from file or profile

    Args:
        config_path: Optional path to configuration file
        profile: Optional profile name ("fast", "hybrid", "quality", "cpu")

    Returns:
        Loaded configuration

    Example:
        >>> config = load_config(profile="fast")
        >>> config = load_config(config_path=Path("configs/custom.yaml"))
    """
    if config_path:
        manager = ConfigManager.from_file(config_path)
    elif profile:
        manager = ConfigManager.from_profile(profile)
    else:
        # Default to fast profile
        manager = ConfigManager.from_profile("fast")

    return manager.get_config()


def create_default_config(output_path: Path, profile: str = "fast") -> None:
    """
    Create default configuration file

    Args:
        output_path: Path to save configuration file
        profile: Profile name to use as template

    Example:
        >>> create_default_config(Path("configs/default.yaml"), profile="fast")
    """
    manager = ConfigManager.from_profile(profile)
    manager.save(output_path)
