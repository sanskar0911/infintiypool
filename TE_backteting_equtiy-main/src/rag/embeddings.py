"""
embeddings.py

Generates vector embeddings for RAG retrieval using SentenceTransformers or TF-IDF fallback.
"""

import importlib.util
from typing import List
import numpy as np

HAS_SENTENCE_TRANSFORMERS = False
_MODEL = None

if importlib.util.find_spec("sentence_transformers") is not None:
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        HAS_SENTENCE_TRANSFORMERS = True
    except Exception:
        HAS_SENTENCE_TRANSFORMERS = False
        _MODEL = None


class EmbeddingEngine:
    """Embedding engine with SentenceTransformers or TF-IDF fallback."""

    def __init__(self):
        self.use_st = HAS_SENTENCE_TRANSFORMERS and (_MODEL is not None)
        if not self.use_st:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer on corpus and transform texts into dense float32 arrays."""
        if self.use_st and _MODEL is not None:
            embeddings = _MODEL.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english")
            sparse_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            dense = sparse_matrix.toarray().astype(np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense / norms

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform query texts into embedding vectors."""
        if self.use_st and _MODEL is not None:
            embeddings = _MODEL.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            if not getattr(self, "is_fitted", False):
                raise ValueError("TF-IDF Vectorizer must be fitted before transforming queries.")
            sparse_matrix = self.vectorizer.transform(texts)
            dense = sparse_matrix.toarray().astype(np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense / norms
