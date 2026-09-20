"""
translator package inside src/rag/translator

Multilingual Language Detection and MADRID Intermediate Representation System.
"""

from rag.translator.schemas import (
    LanguageDetectionResult,
    MADRIDTranslationResult,
    TokenStats,
    RAGPipelineResult
)
from rag.translator.detector import detect_language
from rag.translator.madrid_encoder import to_madrid
from rag.translator.madrid_decoder import from_madrid
from rag.translator.translator import MultilingualMADRIDTranslator
from rag.translator.token_counter import count_tokens, calculate_token_stats

__all__ = [
    "LanguageDetectionResult",
    "MADRIDTranslationResult",
    "TokenStats",
    "RAGPipelineResult",
    "detect_language",
    "to_madrid",
    "from_madrid",
    "MultilingualMADRIDTranslator",
    "count_tokens",
    "calculate_token_stats"
]
