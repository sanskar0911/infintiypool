# Portfolio Construction and Asset Allocation

## Definition
Portfolio construction is the process of combining multiple financial assets or trading strategies into a unified investment portfolio to optimize total risk-adjusted returns.

## Why It Matters
Individual equity strategies often exhibit uncorrelated returns. Combining multiple strategies or stocks reduces overall portfolio variance without sacrificing expected return (Markowitz Modern Portfolio Theory).

## Core Concept
- **Diversification**: Spreading capital across uncorrelated assets.
- **Equal Weighting**: Allocating an equal percentage of capital across selected universe stocks.
- **Risk Parity**: Allocating capital inversely proportional to asset volatility.

## How It Works
1. Define universe of target equities (e.g. INFY, RELIANCE, TCS).
2. Generate signals and calculate asset correlation matrix.
3. Rebalance allocations periodically (e.g. monthly or quarterly).

## Formula
$$\sigma_{\text{portfolio}}^2 = \sum_i w_i^2 \sigma_i^2 + \sum_i \sum_{j \neq i} w_i w_j \sigma_{ij}$$

## Example
Constructing a portfolio of 5 NIFTY stocks allocated at 20% weight each with monthly rebalancing.

## Common Mistakes
- Over-concentrating capital into a single correlated market sector.
- Failing to account for transaction fees during frequent rebalancing.

## Backtesting Considerations
Benchmark portfolio equity against benchmark indices (e.g. NIFTY 50).

## QUANTLAB Implementation
In QUANTLAB, `src/server.py` evaluates multi-metric recommendations across universe stocks and computes benchmark alpha/beta comparison.

## Related Concepts
- Diversification
- Benchmark Comparison
- Volatility
