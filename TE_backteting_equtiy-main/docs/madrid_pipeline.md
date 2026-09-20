# MADRID Translation & Token Accounting Pipeline

## Overview
**MADRID** is an internal canonical intermediate representation used in QUANTLAB to process multilingual quantitative finance queries without compromising domain terminology or introducing machine translation distortions.

## Why MADRID?
Standard natural language translation degrades quantitative terms. For instance, translating "Sharpe Ratio" or "Maximum Drawdown" into literal language equivalents introduces non-standard phrasing that ruins vector search retrieval and LLM reasoning. MADRID canonicalizes queries while preserving key financial terms.

## Terminology Preservation Standard
The following terms remain strictly preserved:
- `Sharpe Ratio`
- `CAGR`
- `Sortino Ratio`
- `Maximum Drawdown`
- `Volatility`
- `Slippage`
- `Transaction Cost`
- `Liquidity`
- `Look-ahead Bias`
- `Survivorship Bias`
- `Point-in-time Correctness`
- `NIFTY 50`
- `SMA` / `EMA` / `RSI` / `OHLCV`

## Scientific Token Accounting
The system tracks exact token usage across 7 dimensions:
1. `Original Input Tokens`
2. `MADRID Input Tokens`
3. `RAG Context Tokens`
4. `LLM Input Tokens`
5. `LLM Output Tokens`
6. `Final Translation Tokens`
7. `Total Tokens`

Observed Token Difference percentage is computed as:
$$\text{Observed Difference \%} = \frac{\text{MADRID Tokens} - \text{Original Tokens}}{\text{Original Tokens}} \times 100$$
