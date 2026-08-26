# RAG Retrieval Evaluation Report

**Milestone Target:** 29-Aug-2026  
**Topic:** Retrieval Quality Benchmarking  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Methodology

The RAG retrieval evaluation module (`src/rag/evaluation.py`) measures the empirical accuracy of the vector retriever against a curated ground-truth dataset of quantitative questions and expected source documents.

### Metrics Defined:
- **Top-1 Accuracy**: Percentage of test questions where the exact ground-truth document is retrieved as the #1 nearest neighbor.
- **Top-3 Accuracy**: Percentage of test questions where the exact ground-truth document appears within the top 3 retrieved results.

---

## 2. Test Dataset & Empirical Benchmark Results

| Test Question | Expected Ground-Truth Source | Retrieved Top-1 Source | Top-1 Hit? | Top-3 Hit? |
|---|---|---|---|---|
| *"What is look-ahead bias in backtesting?"* | `backtesting/look_ahead_bias.md` | `backtesting/look_ahead_bias.md` | YES | YES |
| *"What is Sharpe Ratio and how is it calculated?"* | `finance/sharpe.md` | `finance/sharpe.md` | YES | YES |
| *"How does Simple Moving Average (SMA) crossover work?"* | `strategies/sma.md` | `strategies/sma.md` | YES | YES |
| *"What is maximum drawdown and how to calculate recovery gain?"* | `finance/drawdown.md` | `finance/drawdown.md` | YES | YES |
| *"What causes overfitting in quantitative equity strategies?"* | `backtesting/overfitting.md` | `backtesting/overfitting.md` | YES | YES |
| *"How does Exponential Moving Average (EMA) differ from SMA?"* | `strategies/ema.md` | `strategies/ema.md` | YES | YES |
| *"How do transaction costs and slippage affect backtests?"* | `finance/transaction_costs.md` | `finance/transaction_costs.md` | YES | YES |
| *"What is Relative Strength Index (RSI) mean reversion?"* | `strategies/rsi.md` | `strategies/rsi.md` | YES | YES |

---

## 3. Summary Performance Metrics

- **Total Test Questions**: 8
- **Top-1 Retrieval Hits**: 8 / 8
- **Top-1 Retrieval Accuracy**: **100.0%**
- **Top-3 Retrieval Hits**: 8 / 8
- **Top-3 Retrieval Accuracy**: **100.0%**

> **Note:** Accuracy values are computed empirically dynamically at test runtime by executing `evaluate_rag_retrieval()` against the FAISS vector index.
