"""
vector_store.py

FAISS & NumPy Vector Store index manager supporting index building, loading, updating,
and top-K similarity search with score calculation.
"""

import importlib.util
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

HAS_FAISS = False
faiss_module = None

if importlib.util.find_spec("faiss") is not None:
    try:
        faiss_module = importlib.import_module("faiss")
        HAS_FAISS = True
    except Exception:
        HAS_FAISS = False


class VectorStoreManager:
    """FAISS Vector Index manager with cosine / distance similarity search."""

    def __init__(self):
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Build vector index from document chunk embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)

        if embeddings is None or len(embeddings) == 0:
            return

        dimension = embeddings.shape[1]

        if HAS_FAISS and faiss_module is not None:
            self.index = faiss_module.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            print(f"[FAISS Vector Store] Indexed {self.index.ntotal} vectors with dimension {dimension}.")
        else:
            print(f"[Numpy Vector Store Fallback] Indexed {len(chunks)} vectors with dimension {dimension}.")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search vector store for top_k nearest chunks.

        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            List of (chunk, similarity_score) tuples where similarity_score is between 0.0 and 1.0.
        """
        if self.chunks is None or len(self.chunks) == 0 or self.embeddings is None:
            return []

        q_vec = query_embedding.astype(np.float32)
        if len(q_vec.shape) == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        results: List[Tuple[Dict[str, Any], float]] = []

        if HAS_FAISS and self.index is not None:
            distances, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.chunks):
                    # Convert L2 distance to similarity score range [0, 1]
                    sim = round(max(0.0, 1.0 / (1.0 + float(dist))), 4)
                    results.append((self.chunks[idx], sim))
        else:
            # Cosine / dot-product similarity search for numpy array
            dists = np.linalg.norm(self.embeddings - q_vec, axis=1)
            top_indices = np.argsort(dists)[:top_k]
            for idx in top_indices:
                sim = round(max(0.0, 1.0 / (1.0 + float(dists[idx]))), 4)
                results.append((self.chunks[idx], sim))

        return results

    def clear_index(self):
        """Clear vector index."""
        self.index = None
        self.chunks = []
        self.embeddings = None
