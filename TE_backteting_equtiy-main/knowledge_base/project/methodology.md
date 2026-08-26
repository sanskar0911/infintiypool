# Backtesting Methodology & Standard Operating Procedures

## Methodological Steps
1. **Data Ingestion**: Clean daily OHLCV equity price history with date indexing.
2. **Deterministic Signal Generation**: Compute technical indicators (SMA, EMA, RSI) without future look-ahead.
3. **Execution Simulation**: Execute positions with realistic slippage (0.05%) and commission fees (0.1%).
4. **Performance Evaluation**: Compute Sharpe ratio, CAGR, Maximum Drawdown, and Win Rate.
5. **AI Analysis & Retrieval**: Query FAISS vector index for strategy knowledge and synthesize findings using LLM structured prompts.
