"""
schemas.py inside src/rag/translator

Data structures for language detection, MADRID translation, token statistics, and RAG pipeline outputs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LanguageDetectionResult:
    language_name: str
    language_code: str
    confidence: float = 1.0


@dataclass
class MADRIDTranslationResult:
    original_text: str
    source_language: str
    source_language_code: str
    madrid_text: str
    token_count_original: int
    token_count_madrid: int
    char_count_original: int
    char_count_madrid: int


@dataclass
class TokenStats:
    original_input_tokens: int = 0
    madrid_input_tokens: int = 0
    rag_context_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    final_translation_tokens: int = 0
    total_tokens: int = 0
    observed_token_difference_pct: float = 0.0


@dataclass
class RAGPipelineResult:
    original_query: str
    detected_language: str
    language_code: str
    madrid_query: str
    retrieved_sources: List[Dict[str, Any]] = field(default_factory=list)
    madrid_response: str = ""
    final_answer: str = ""
    token_stats: Dict[str, Any] = field(default_factory=dict)
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
