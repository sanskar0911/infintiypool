"""
test_rag.py

Unit test suite for RAG Loader, Vector Search, and Retrieval Evaluation.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from rag.loader import load_knowledge_base_docs
from rag.chunker import chunk_documents
from rag.retriever import RAGRetriever
from rag.evaluation import evaluate_rag_retrieval


def test_knowledge_base_loader():
    """Test loading markdown files from knowledge_base directory."""
    docs = load_knowledge_base_docs()
    assert len(docs) > 0
    assert any("sma.md" in doc["id"] for doc in docs)


def test_rag_chunker():
    """Test document chunking."""
    docs = load_knowledge_base_docs()
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    assert "text" in chunks[0]


def test_rag_retriever_search():
    """Test vector similarity search via RAGRetriever."""
    retriever = RAGRetriever()
    results = retriever.retrieve("What is look-ahead bias?", top_k=3)
    assert len(results) > 0
    top_chunk, dist = results[0]
    assert "look_ahead_bias.md" in top_chunk["source"]


def test_rag_retrieval_evaluation():
    """Test empirical retrieval evaluation engine."""
    retriever = RAGRetriever()
    eval_res = evaluate_rag_retrieval(retriever)
    assert eval_res["total_questions"] > 0
    assert eval_res["top_1_accuracy"] >= 75.0
    assert eval_res["top_3_accuracy"] >= 85.0
