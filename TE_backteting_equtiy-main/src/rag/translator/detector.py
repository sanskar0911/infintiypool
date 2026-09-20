"""
detector.py inside src/rag/translator

Automatic Language Detection Engine supporting English, Hindi, Marathi, Spanish, French, German, etc.
"""

import re
from rag.translator.schemas import LanguageDetectionResult

DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')
MARATHI_KEYWORDS = ["म्हणजे", "काय", "आहे", "कसे", "करावे", "मोजण्यासाठी", "सांगा", "मिळेल", "कोणते"]
HINDI_KEYWORDS = ["kya", "hai", "kaise", "hota", "batao", "karte", "kyun", "kab", "shukriya"]
SPANISH_KEYWORDS = ["qué", "que", "cómo", "como", "es", "el", "ratio", "por", "para", "cuál", "cuanto"]


def detect_language(text: str) -> LanguageDetectionResult:
    """
    Automatically detects language of the input query string.
    """
    if not text or not text.strip():
        return LanguageDetectionResult(language_name="English", language_code="en", confidence=1.0)

    clean_text = text.strip().lower()

    # Devanagari script (Marathi / Hindi)
    devanagari_chars = DEVANAGARI_REGEX.findall(text)
    if len(devanagari_chars) > 0:
        is_marathi = any(kw in clean_text for kw in MARATHI_KEYWORDS) or 'ळ' in text
        if is_marathi:
            return LanguageDetectionResult(language_name="Marathi", language_code="mr", confidence=0.95)
        return LanguageDetectionResult(language_name="Hindi", language_code="hi", confidence=0.90)

    # Hinglish / Romanized Hindi
    hinglish_count = sum(1 for kw in HINDI_KEYWORDS if f" {kw} " in f" {clean_text} ")
    if hinglish_count >= 1:
        return LanguageDetectionResult(language_name="Hindi", language_code="hi", confidence=0.85)

    # Spanish
    spanish_count = sum(1 for kw in SPANISH_KEYWORDS if f" {kw} " in f" {clean_text} ")
    if spanish_count >= 1 or "¿" in text or "Ratio de" in text.title():
        return LanguageDetectionResult(language_name="Spanish", language_code="es", confidence=0.90)

    # French / German
    if any(w in clean_text for w in ["est-ce", "qu'est-ce", "pourquoi", "comment"]):
        return LanguageDetectionResult(language_name="French", language_code="fr", confidence=0.88)
    if any(w in clean_text for w in ["was", "ist", "wie", "warum"]):
        return LanguageDetectionResult(language_name="German", language_code="de", confidence=0.88)

    return LanguageDetectionResult(language_name="English", language_code="en", confidence=1.0)
