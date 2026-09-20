# Retrieval-Augmented Generation (RAG) in Quantitative Research

## Definition
Retrieval-Augmented Generation (RAG) is an AI architecture that enhances Large Language Models (LLMs) by retrieving relevant factual context chunks from an external knowledge base prior to generating responses.

## Why It Matters
LLMs suffer from hallucinations and lack domain-specific or point-in-time quantitative context. RAG grounds AI responses in verified financial documentation, preventing fabricated metrics and outdated rules.

## Core Concept
1. **Indexing**: Document loading, cleaning, chunking, and vector embedding.
2. **Retrieval**: Nearest-neighbor vector similarity search over query embeddings.
3. **Synthesis**: LLM prompt enrichment using retrieved context chunks.

## How It Works
$$\text{User Query} \xrightarrow{\text{Embedding}} \text{Vector Index} \xrightarrow{\text{Top-K Chunks}} \text{Context} + \text{Query} \xrightarrow{\text{LLM}} \text{Grounded Answer}$$

## Formula
$$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

## Example
When asked about Sharpe Ratio calculation, RAG retrieves `03_metrics/sharpe_ratio.md` and feeds the exact formula into the LLM prompt.

## Common Mistakes
- Chunking documents without preserving section metadata or source paths.
- Allowing the LLM to generate ungrounded claims when retrieval similarity scores are low.

## Backtesting Considerations
Retrieval evaluation metrics (Top-1, Top-3 accuracy) monitor knowledge base retrieval quality.

## QUANTLAB Implementation
Implemented in `src/rag/` modules and visual pipeline in `src/server.py`.

## Related Concepts
- MADRID Pipeline
- Vector Store & FAISS
- Hallucination Control
