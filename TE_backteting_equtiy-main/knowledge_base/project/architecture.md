# Platform Architecture Overview

## Shankh / Decuple Internship 2026-27 Project TE1 Architecture

The platform follows a modular Python architecture comprising 5 primary layers:

1. **Core Data & Backtest Engine**: Ingests stock price CSVs, generates signals (SMA, EMA, RSI), and executes simulation loops with friction models.
2. **Benchmark Engine**: Compares equity curves against Nifty 50 historical benchmark returns.
3. **Structured LLM Foundation**: Binds calculated metrics into institutional JSON evaluation reports using OpenAI/Gemini APIs.
4. **LangGraph Agent & Loop Engineering**: Multi-node state graph orchestrating tools and executing controlled parameter iterations.
5. **FAISS RAG Research Assistant**: Semantic vector retriever and Q&A engine backed by markdown documentation corpus.
