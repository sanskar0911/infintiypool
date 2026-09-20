"""
test_madrid_pipeline.py

Pytest test suite for Language Detection, MADRID Encoding/Decoding, Token Accounting,
and End-to-End RAG Pipeline execution.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from translator.detector import detect_language
from translator.madrid_encoder import to_madrid
from translator.madrid_decoder import from_madrid
from translator.token_counter import count_tokens, calculate_token_stats
from translator.translator import MultilingualMADRIDTranslator
from rag.rag_chain import RAGChain


def test_language_detection():
    """Test automatic language detection for EN, HI, MR, ES."""
    res_en = detect_language("What is Sharpe ratio?")
    assert res_en.language_code == "en"

    res_mr = detect_language("Sharpe ratio म्हणजे काय?")
    assert res_mr.language_code == "mr"

    res_hi = detect_language("Sharpe ratio kya hota hai?")
    assert res_hi.language_code == "hi"

    res_es = detect_language("¿Qué es el ratio de Sharpe?")
    assert res_es.language_code == "es"


def test_madrid_encoding_terminology_preservation():
    """Verify quantitative terms remain preserved in MADRID encoding."""
    query = "Sharpe ratio आणि Maximum Drawdown म्हणजे काय?"
    res = to_madrid(query, source_language_code="mr")
    assert "Sharpe" in res.madrid_text or "Sharpe ratio" in res.madrid_text
    assert "Drawdown" in res.madrid_text or "Maximum Drawdown" in res.madrid_text


def test_madrid_decoding():
    """Verify MADRID response decoding into target language."""
    decoded = from_madrid("Sharpe ratio is a risk metric.", target_language_code="mr")
    assert "Sharpe Ratio" in decoded


def test_token_accounting():
    """Verify token stats calculation."""
    stats = calculate_token_stats(
        original_input="What is Sharpe ratio?",
        madrid_input="[MADRID::...] What is Sharpe ratio?",
        rag_context="Sharpe ratio formula context...",
        llm_output_madrid="Sharpe ratio answer...",
        final_output="Sharpe ratio is..."
    )
    assert stats["original_input_tokens"] > 0
    assert stats["madrid_input_tokens"] > 0
    assert stats["total_tokens"] > 0
    assert "observed_token_difference_pct" in stats


def test_rag_pipeline_execution():
    """Verify end-to-end RAG execution for English and Marathi queries."""
    chain = RAGChain()
    
    # Test English query
    res_en = chain.execute_pipeline("What is Sharpe ratio?", top_k=2)
    assert res_en["status"] in ["success", "grounding_insufficient"]
    assert "token_stats" in res_en

    # Test Marathi query
    res_mr = chain.execute_pipeline("Sharpe ratio म्हणजे काय?", top_k=2)
    assert res_mr["detected_language"] == "Marathi"
    assert res_mr["language_code"] == "mr"
    assert "final_answer" in res_mr
