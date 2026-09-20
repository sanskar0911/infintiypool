# QUANTLAB Architecture & Multilingual MADRID Pipeline

## Definition
QUANTLAB is an AI-powered quantitative equity backtesting, research, and analysis platform featuring a 8-stage Multilingual RAG Pipeline powered by MADRID (Multilingual Intermediate Canonical Representation).

## Why It Matters
Standard LLM wrappers fail at multilingual quantitative research because raw machine translation degrades technical financial metrics (e.g. converting "Sharpe Ratio" or "Maximum Drawdown" into literal non-standard phrases). MADRID canonicalizes queries to preserve technical domain precision and enable transparent token accounting.

## Core Concept
The 8-stage observable pipeline flow:
1. **User Input**: Query ingestion across any natural language (EN, HI, MR, ES, etc.).
2. **Language Detection**: Automatic language & script identifier.
3. **MADRID Translation**: Ingestion into canonical intermediate representation (`[MADRID::...]`).
4. **RAG Retrieval**: Top-K vector store similarity search across QUANTLAB knowledge base.
5. **Retrieved Knowledge**: Filtered context chunks with similarity scores & source metadata.
6. **LLM Synthesis**: Grounded quantitative analysis using MADRID query + context.
7. **MADRID Decoder**: Target language translation with strict terminology preservation.
8. **Final User Output**: Formatted response, source citations, latency, and token breakdown.

## How It Works
1. `translator/detector.py` identifies language code.
2. `translator/madrid_encoder.py` builds canonical MADRID string.
3. `src/rag/retriever.py` searches vector store.
4. `src/rag/rag_chain.py` generates grounded answer.
5. `translator/madrid_decoder.py` returns target language response.
6. `translator/token_counter.py` measures token usage across all steps.

## Formula
$$\text{Observed Token Difference \%} = \frac{\text{MADRID Input Tokens} - \text{Original Input Tokens}}{\text{Original Input Tokens}} \times 100$$

## Example
Input: `"Sharpe ratio म्हणजे काय?"` $\rightarrow$ Detected: Marathi (mr) $\rightarrow$ MADRID Query: `[MADRID::...]` $\rightarrow$ RAG Search: `03_metrics/sharpe_ratio.md` (Score: 0.91) $\rightarrow$ Final Output: `"Sharpe Ratio हा risk-adjusted return मोजण्यासाठी..."`.

## Common Mistakes
- Blindly assuming non-English machine translation saves tokens without measuring actual subword tokenization.

## Backtesting Considerations
Integrates cleanly with agent tools and backtesting modules without coupling core evaluation logic.

## QUANTLAB Implementation
Core modules reside in `translator/`, `src/rag/`, and served visually via `src/server.py`.

## Related Concepts
- Grounded RAG
- Token Accounting
- LangGraph Workflow
