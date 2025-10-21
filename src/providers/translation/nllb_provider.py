"""
NLLB (No Language Left Behind) Translation Provider

Local multilingual translation using Meta's NLLB-200 models.
Supports 200+ languages without API costs.
"""

import time
import torch
from typing import AsyncIterator, Optional, Dict, Any
from collections import defaultdict

from src.providers.base import (
    TranslationProvider,
    TranslationResult,
    ProviderStatus,
)


# Language code mapping: common codes -> NLLB codes
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "zh-CN": "zho_Hans",
    "zh-TW": "zho_Hant",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
}


class NLLBProvider(TranslationProvider):
    """
    NLLB-200 translation provider for local multilingual translation

    Features:
    - 200+ languages support
    - No API costs (local execution)
    - Fast inference on GPU
    - Batch translation support
    - Optional translation caching
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NLLB provider

        Args:
            config: Configuration dictionary with:
                - model: Model name (default: facebook/nllb-200-distilled-1.3B)
                - device: Device to use (cpu, cuda, auto)
                - precision: Model precision (float16, float32)
                - max_length: Maximum sequence length (default: 512)
                - num_beams: Beam search size (default: 4)
                - cache_enabled: Enable translation caching (default: True)
                - cache_size: Max cache entries (default: 10000)
        """
        super().__init__(config)

        self.model_name = config.get("model", "facebook/nllb-200-distilled-1.3B")
        self.device = config.get("device", "auto")
        self.precision = config.get("precision", "float16")
        self.max_length = config.get("max_length", 512)
        self.num_beams = config.get("num_beams", 4)

        # Caching
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_size = config.get("cache_size", 10000)
        self._cache: Dict[tuple, str] = {}

        # Model and tokenizer (initialized in initialize())
        self.model = None
        self.tokenizer = None

        # Supported languages (subset of NLLB-200)
        self._supported_languages = list(LANG_CODE_MAP.keys())

    async def initialize(self) -> None:
        """Initialize NLLB model and tokenizer"""
        if self.status == ProviderStatus.READY:
            return

        self.status = ProviderStatus.INITIALIZING

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            # Auto-detect device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )

            # Load model with safetensors (safer and faster)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.precision == "float16" and self.device == "cuda" else torch.float32,
                use_safetensors=True,  # Force safetensors format to avoid CVE-2025-32434
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            self.status = ProviderStatus.READY

        except ImportError as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(
                "transformers not installed. Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(f"Failed to initialize NLLB: {e}") from e

    async def cleanup(self) -> None:
        """Clean up NLLB model"""
        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        self._cache.clear()
        self.status = ProviderStatus.STOPPED

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text from source to target language

        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "zh", "en")
            target_lang: Target language code
            context: Optional context (not used in NLLB)

        Returns:
            Translation result with translated text
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0,
                timestamp=time.time(),
                model_name=self.model_name,
            )

        start_time = time.time()

        try:
            # Check cache
            cache_key = (text, source_lang, target_lang)
            if self.cache_enabled and cache_key in self._cache:
                translated_text = self._cache[cache_key]
                duration_ms = (time.time() - start_time) * 1000
                self._record_request(duration_ms, error=False)

                return TranslationResult(
                    original_text=text,
                    translated_text=translated_text,
                    source_language=source_lang,
                    target_language=target_lang,
                    confidence=1.0,  # Cached result
                    timestamp=time.time(),
                    model_name=self.model_name,
                )

            # Convert language codes to NLLB format
            src_lang_code = LANG_CODE_MAP.get(source_lang, source_lang)
            tgt_lang_code = LANG_CODE_MAP.get(target_lang, target_lang)

            # Tokenize with source language
            self.tokenizer.src_lang = src_lang_code
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang_code),
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=False,
                )

            # Decode translation
            translated_text = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0]

            # Cache result
            if self.cache_enabled:
                if len(self._cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = translated_text

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            self._record_request(duration_ms, error=False)

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.95,  # NLLB doesn't provide confidence scores
                timestamp=time.time(),
                model_name=self.model_name,
            )

        except Exception as e:
            self._record_request(0, error=True)
            raise RuntimeError(f"Translation failed: {e}") from e

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts in batch (more efficient)

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translation results
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        if not texts:
            return []

        start_time = time.time()

        try:
            # Convert language codes
            src_lang_code = LANG_CODE_MAP.get(source_lang, source_lang)
            tgt_lang_code = LANG_CODE_MAP.get(target_lang, target_lang)

            # Check cache and separate cached/uncached texts
            results = [None] * len(texts)
            uncached_indices = []
            uncached_texts = []

            for i, text in enumerate(texts):
                if not text or not text.strip():
                    results[i] = TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_language=source_lang,
                        target_language=target_lang,
                        confidence=1.0,
                        timestamp=time.time(),
                        model_name=self.model_name,
                    )
                    continue

                cache_key = (text, source_lang, target_lang)
                if self.cache_enabled and cache_key in self._cache:
                    results[i] = TranslationResult(
                        original_text=text,
                        translated_text=self._cache[cache_key],
                        source_language=source_lang,
                        target_language=target_lang,
                        confidence=1.0,
                        timestamp=time.time(),
                        model_name=self.model_name,
                    )
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            # Batch translate uncached texts
            if uncached_texts:
                self.tokenizer.src_lang = src_lang_code
                inputs = self.tokenizer(
                    uncached_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang_code),
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        do_sample=False,
                    )

                translated_texts = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )

                # Fill results and cache
                for i, idx in enumerate(uncached_indices):
                    original_text = texts[idx]
                    translated_text = translated_texts[i]

                    results[idx] = TranslationResult(
                        original_text=original_text,
                        translated_text=translated_text,
                        source_language=source_lang,
                        target_language=target_lang,
                        confidence=0.95,
                        timestamp=time.time(),
                        model_name=self.model_name,
                    )

                    # Cache
                    if self.cache_enabled:
                        cache_key = (original_text, source_lang, target_lang)
                        if len(self._cache) >= self.cache_size:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[cache_key] = translated_text

            duration_ms = (time.time() - start_time) * 1000
            self._record_request(duration_ms, error=False)

            return results

        except Exception as e:
            self._record_request(0, error=True)
            raise RuntimeError(f"Batch translation failed: {e}") from e

    def supports_batch(self) -> bool:
        """NLLB supports efficient batch translation"""
        return True

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return self._supported_languages

    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        if self.status != ProviderStatus.READY:
            return False

        if self.model is None or self.tokenizer is None:
            return False

        # Test translation with simple text
        try:
            result = await self.translate("Hello", "en", "zh")
            return len(result.translated_text) > 0
        except Exception:
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "enabled": self.cache_enabled,
            "size": len(self._cache),
            "max_size": self.cache_size,
            "hit_rate": getattr(self, "_cache_hits", 0) / max(getattr(self, "_cache_requests", 1), 1),
        }

    def clear_cache(self) -> None:
        """Clear translation cache"""
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"NLLBProvider(model={self.model_name}, "
            f"device={self.device}, precision={self.precision}, "
            f"status={self.status.value})"
        )


# Register provider
from src.providers.base import register_provider
register_provider("translation", "nllb_local", NLLBProvider)
