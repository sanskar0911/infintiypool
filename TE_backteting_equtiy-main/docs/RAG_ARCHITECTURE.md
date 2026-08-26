# RAG Architecture & Vector Store Implementation

**Milestone Target:** 29-Aug-2026 (Current Major Milestone)  
**Topic:** Retrieval-Augmented Generation (RAG, Vector Stores, Memory)  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. RAG System Architecture

The RAG Assistant (`src/rag/`) enables grounding LLM quantitative answers in an authoritative markdown domain knowledge corpus (`knowledge_base/`).

```text
Markdown Knowledge Base (knowledge_base/)
  ├── strategies/ (sma.md, ema.md, rsi.md)
  ├── finance/ (sharpe.md, drawdown.md, transaction_costs.md)
  ├── backtesting/ (look_ahead_bias.md, overfitting.md)
  └── project/ (architecture.md, methodology.md)
               │
               ▼
   Document Loader (src/rag/loader.py)
               │
               ▼
   Text Chunker (src/rag/chunker.py)
[Splits docs by headers into 500-char chunks]
               │
               ▼
 Embedding Engine (src/rag/embeddings.py)
[SentenceTransformers all-MiniLM-L6-v2 / TF-IDF]
               │
               ▼
  FAISS Vector Store (src/rag/vector_store.py)
[faiss.IndexFlatL2 similarity search]
               │
               ▼
   Retriever & RAG Chain (src/rag/rag_chain.py)
[Combines retrieved context + LLM prompt]
               │
               ▼
      Grounded Output with Citations
```

---

## 2. Component Specifications

### 1. Document Loader (`src/rag/loader.py`)
Recursively loads markdown files from `knowledge_base/` and assigns relative source tags (`strategies/sma.md`, `finance/sharpe.md`).

### 2. Header-Aware Text Chunker (`src/rag/chunker.py`)
Splits documents along markdown section headers (`##` / `###`) with paragraph overflow protection (max 500 characters per chunk).

### 3. Embedding Vectorizer (`src/rag/embeddings.py`)
Encodes text chunks into dense float32 vector space using `SentenceTransformers` (`all-MiniLM-L6-v2`). Includes an fallback vectorizer using TF-IDF when dense neural models are offline.

### 4. FAISS Vector Store (`src/rag/vector_store.py`)
Manages FAISS flat L2 index (`faiss.IndexFlatL2`) for fast, exact Euclidean vector similarity search.

### 5. Grounded Q&A Chain (`src/rag/rag_chain.py`)
Binds top-$k$ retrieved context blocks to the user query and generates an answer strictly citing source documents.
