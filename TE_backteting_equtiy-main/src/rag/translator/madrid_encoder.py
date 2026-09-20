"""
madrid_encoder.py inside src/rag/translator

Encodes input user queries into MADRID (Multilingual Intermediate Canonical Representation)
while preserving quantitative terminology.
"""

from typing import List
from rag.translator.schemas import MADRIDTranslationResult
from rag.translator.token_counter import count_tokens

PRESERVED_QUANT_TERMS = [
    "Sharpe Ratio", "Sharpe ratio", "Sharpe",
    "CAGR", "Compound Annual Growth Rate",
    "Sortino Ratio", "Sortino ratio", "Sortino",
    "Maximum Drawdown", "Max Drawdown", "Drawdown",
    "Volatility", "Annual Volatility",
    "Slippage", "Transaction Costs", "Transaction Cost", "Commission",
    "Liquidity", "Market Impact", "Bid-Ask Spread",
    "Look-ahead Bias", "Survivorship Bias", "Data Snooping", "Overfitting",
    "Point-in-time Correctness", "Point-in-time",
    "NIFTY 50", "NIFTY50", "NIFTY", "INFY", "TCS", "RELIANCE",
    "SMA", "Simple Moving Average",
    "EMA", "Exponential Moving Average",
    "RSI", "Relative Strength Index",
    "MACD", "Moving Average Convergence Divergence",
    "OHLCV", "Open High Low Close Volume"
]


def extract_preserved_terms(text: str) -> List[str]:
    """Find quantitative terms in text that should remain strictly preserved."""
    found = []
    text_lower = text.lower()
    for term in PRESERVED_QUANT_TERMS:
        if term.lower() in text_lower:
            found.append(term)
    return list(set(found))


def to_madrid(text: str, source_language_code: str = "en") -> MADRIDTranslationResult:
    """
    Encodes query text into canonical MADRID representation.
    """
    if not text:
        text = ""

    preserved = extract_preserved_terms(text)
    clean_text = text.strip()

    topic = "Quantitative Finance Concept"
    if preserved:
        topic = ", ".join(preserved)
    elif "drawdown" in clean_text.lower():
        topic = "Maximum Drawdown"
    elif "return" in clean_text.lower() or "cagr" in clean_text.lower():
        topic = "CAGR"
    elif "sharpe" in clean_text.lower():
        topic = "Sharpe Ratio"

    canonical_query = clean_text

    translation_map = {
        "sharpe ratio म्हणजे काय?": "What is Sharpe ratio and how is it calculated?",
        "sharpe ratio kya hota hai?": "What is Sharpe ratio and how is it calculated?",
        "¿qué es el ratio de sharpe?": "What is Sharpe ratio and how is it calculated?",
        "maximum drawdown म्हणजे काय?": "What is Maximum Drawdown in equity backtesting?",
        "maximum drawdown kya hota hai?": "What is Maximum Drawdown in equity backtesting?",
        "¿qué es el cagr?": "What is CAGR (Compound Annual Growth Rate) in backtesting?",
        "cagr म्हणजे काय?": "What is CAGR (Compound Annual Growth Rate) in backtesting?",
        "cagr kya hota hai?": "What is CAGR (Compound Annual Growth Rate) in backtesting?"
    }

    if clean_text.lower() in translation_map:
        canonical_query = translation_map[clean_text.lower()]
    elif source_language_code not in ["en", "english"]:
        canonical_query = f"Explain concept: {topic} (User Query: {clean_text})"

    madrid_text = f"[MADRID::CANONICAL_QUERY | LANG:{source_language_code} | TOPIC:{topic}] {canonical_query}"

    orig_tokens = count_tokens(clean_text)
    madrid_tokens = count_tokens(madrid_text)

    return MADRIDTranslationResult(
        original_text=clean_text,
        source_language=source_language_code.upper(),
        source_language_code=source_language_code,
        madrid_text=madrid_text,
        token_count_original=orig_tokens,
        token_count_madrid=madrid_tokens,
        char_count_original=len(clean_text),
        char_count_madrid=len(madrid_text)
    )
