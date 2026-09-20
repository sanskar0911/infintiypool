"""
translator.py

Unified Multilingual MADRID Translation Coordinator providing clean API for language detection,
encoding to MADRID, decoding from MADRID, and calculating token metrics.
"""

from typing import Dict, Any
from translator.schemas import LanguageDetectionResult, MADRIDTranslationResult
from translator.detector import detect_language
from translator.madrid_encoder import to_madrid
from translator.madrid_decoder import from_madrid
from translator.token_counter import count_tokens, calculate_token_stats


class MultilingualMADRIDTranslator:
    """Coordinator class for MADRID multilingual processing."""

    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect language of user input query."""
        return detect_language(text)

    def encode(self, text: str, source_language_code: str = None) -> MADRIDTranslationResult:
        """Encode input text into canonical MADRID representation."""
        if source_language_code is None:
            detected = self.detect(text)
            source_language_code = detected.language_code
        return to_madrid(text, source_language_code=source_language_code)

    def decode(self, madrid_response: str, target_language_code: str = "en") -> str:
        """Decode MADRID response into target user language."""
        return from_madrid(madrid_response, target_language_code=target_language_code)

    def get_token_breakdown(
        self,
        original_input: str,
        madrid_input: str,
        rag_context: str,
        llm_output_madrid: str,
        final_output: str
    ) -> Dict[str, Any]:
        """Compute full token stats for RAG + MADRID pipeline."""
        return calculate_token_stats(
            original_input=original_input,
            madrid_input=madrid_input,
            rag_context=rag_context,
            llm_output_madrid=llm_output_madrid,
            final_output=final_output
        )
