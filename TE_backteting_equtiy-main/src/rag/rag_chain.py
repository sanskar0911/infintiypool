"""
rag_chain.py

Grounded RAG Q&A chain combining semantic search context with LLM synthesis.
"""

import os
import sys
import json
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from rag.retriever import RAGRetriever


class RAGChain:
    """RAG Q&A Chain executing grounded search and LLM context synthesis."""

    def __init__(self, retriever: RAGRetriever = None):
        self.retriever = retriever or RAGRetriever()
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve context and generate grounded answer for user question.

        Returns
        -------
        Dict[str, Any]
            { 'query': str, 'answer': str, 'sources': List[str], 'retrieved_chunks': List[Dict] }
        """
        results = self.retriever.retrieve(query, top_k=top_k)

        if not results:
            return {
                "query": query,
                "answer": "No relevant documents found in knowledge base.",
                "sources": [],
                "retrieved_chunks": []
            }

        retrieved_chunks = [item[0] for item in results]
        sources = list(set([c["source"] for c in retrieved_chunks]))

        context_str = "\n\n---\n\n".join([f"Source: {c['source']}\nContent:\n{c['text']}" for c in retrieved_chunks])

        prompt = (
            f"You are an expert Quantitative Finance & Backtesting AI Assistant.\n"
            f"Answer the user query strictly using the provided Knowledge Base Context.\n"
            f"If the context does not contain enough details, answer based on quantitative best practices and cite the relevant source files.\n\n"
            f"User Question: {query}\n\n"
            f"Knowledge Base Context:\n{context_str}\n\n"
            f"Provide a clear, thorough, and professional answer."
        )

        # 1. Attempt OpenAI API
        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a quantitative research assistant. Answer accurately based on context."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer_text = response.choices[0].message.content
                return {
                    "query": query,
                    "answer": answer_text,
                    "sources": sources,
                    "retrieved_chunks": retrieved_chunks
                }
            except Exception:
                pass

        # 2. Grounded Fallback Answer Synthesis combining top chunks
        chunk_sections = []
        for idx, c in enumerate(retrieved_chunks, 1):
            chunk_sections.append(f"**Section {idx} (from `{c['source']}`):**\n{c['text']}")

        combined_context = "\n\n".join(chunk_sections)
        answer_text = (
            f"Based on quantitative documentation from the repository:\n\n"
            f"{combined_context}\n\n"
            f"**Sources Consulted**: {', '.join([f'`{s}`' for s in sources])}"
        )

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }
