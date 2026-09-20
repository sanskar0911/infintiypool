"""
retriever.py

High-level RAG semantic retriever coordinating vector search over knowledge base.
Includes singleton instance management to prevent redundant indexing.
"""

from typing import List, Dict, Any, Tuple
from rag.loader import load_knowledge_base_docs
from rag.chunker import chunk_documents
from rag.embeddings import EmbeddingEngine
from rag.vector_store import VectorStoreManager

_GLOBAL_RETRIEVER = None


class RAGRetriever:
    """RAG semantic retriever managing vector indexing and document retrieval."""

    def __init__(self, kb_dir: str = None, chunk_size: int = 500, chunk_overlap: int = 50):
        self.loader = load_knowledge_base_docs(kb_dir)
        self.chunks = chunk_documents(self.loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingEngine()
        self.vector_store = VectorStoreManager()

        if self.chunks:
            chunk_texts = [c["text"] for c in self.chunks]
            embeddings = self.embedder.fit_transform(chunk_texts)
            self.vector_store.build_index(self.chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 3, similarity_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top_k most relevant knowledge base chunks for user query.

        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            List of (chunk_dict, similarity_score) tuples.
        """
        if not self.chunks:
            return []

        q_embed = self.embedder.transform([query])
        results = self.vector_store.search(q_embed, top_k=top_k)

        if similarity_threshold > 0.0:
            results = [r for r in results if r[1] >= similarity_threshold]

        return results


def get_retriever(kb_dir: str = None) -> RAGRetriever:
    """Get or initialize singleton RAGRetriever instance."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = RAGRetriever(kb_dir=kb_dir)
    return _GLOBAL_RETRIEVER
