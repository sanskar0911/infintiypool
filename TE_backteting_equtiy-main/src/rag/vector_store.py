"""
vector_store.py

FAISS Vector Store index manager for semantic document retrieval.
"""

import importlib.util
from typing import List, Dict, Any, Tuple
import numpy as np

HAS_FAISS = False
faiss_module = None

if importlib.util.find_spec("faiss") is not None:
    try:
        import faiss as faiss_module
        HAS_FAISS = True
    except Exception:
        HAS_FAISS = False


class VectorStoreManager:
    """FAISS Vector Index manager with L2 distance similarity search."""

    def __init__(self):
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Build FAISS index from document chunk embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)

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
            List of (chunk, distance_score) tuples.
        """
        if self.chunks is None or len(self.chunks) == 0:
            return []

        q_vec = query_embedding.astype(np.float32)
        if len(q_vec.shape) == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        results: List[Tuple[Dict[str, Any], float]] = []

        if HAS_FAISS and self.index is not None:
            distances, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(dist)))
        else:
            dists = np.linalg.norm(self.embeddings - q_vec, axis=1)
            top_indices = np.argsort(dists)[:top_k]
            for idx in top_indices:
                results.append((self.chunks[idx], float(dists[idx])))

        return results
