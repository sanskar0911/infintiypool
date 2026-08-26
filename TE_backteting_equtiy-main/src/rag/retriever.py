"""
retriever.py

High-level RAG semantic retriever coordinating vector search over the knowledge base.
"""

from typing import List, Dict, Any, Tuple
from rag.loader import load_knowledge_base_docs
from rag.chunker import chunk_documents
from rag.embeddings import EmbeddingEngine
from rag.vector_store import VectorStoreManager


class RAGRetriever:
    """RAG semantic retriever managing vector indexing and document retrieval."""

    def __init__(self, kb_dir: str = None):
        self.loader = load_knowledge_base_docs(kb_dir)
        self.chunks = chunk_documents(self.loader)
        self.embedder = EmbeddingEngine()
        self.vector_store = VectorStoreManager()

        if self.chunks:
            chunk_texts = [c["text"] for c in self.chunks]
            embeddings = self.embedder.fit_transform(chunk_texts)
            self.vector_store.build_index(self.chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top_k most relevant knowledge base chunks for user query."""
        if not self.chunks:
            return []

        q_embed = self.embedder.transform([query])
        return self.vector_store.search(q_embed, top_k=top_k)
