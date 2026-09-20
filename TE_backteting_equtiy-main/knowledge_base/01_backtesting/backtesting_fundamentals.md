# Backtesting Fundamentals

## Definition
Backtesting is the process of testing a trading strategy or mathematical model on historical financial market data to evaluate its hypothetical performance, risk metrics, and viability before committing real capital.

## Why It Matters
Without rigorous backtesting, quantitative traders risk deploying strategies that are unprofitable, excessively volatile, or prone to catastrophic drawdowns. Backtesting validates trading logic against past market cycles.

## Core Concept
The core concept is simulating trade execution chronologically line-by-line across historical OHLCV bar data, tracking position state, available cash, cumulative transaction costs, and portfolio value.

## How It Works
1. Historical price series (Open, High, Low, Close, Volume) is ingested.
2. Signal generation functions evaluate indicator rules to issue BUY (+1), SELL (-1), or HOLD (0) signals.
3. The execution engine computes trades, applying commission and slippage models.
4. Equity curves and risk metrics (Sharpe Ratio, CAGR, Max Drawdown) are generated.

## Formula
$$\text{Portfolio Value}_t = \text{Cash}_t + \sum (\text{Shares}_{i,t} \times \text{Price}_{i,t})$$

## Example
A strategy buys 100 shares of INFY at ₹1,500 on a 20-day SMA crossover and sells at ₹1,650 when the short SMA crosses below the long SMA, yielding a gross profit of ₹15,000 minus transaction costs.

## Common Mistakes
- Including future price data in signal calculation (Look-ahead bias).
- Ignoring transaction costs, commission rates, and market slippage.
- Testing on survivor-only universes (Survivorship bias).

## Backtesting Considerations
Always ensure point-in-time correctness and model realistic market friction parameters.

## QUANTLAB Implementation
In QUANTLAB, `src/backtester.py` executes full vectorised and event-driven backtesting loops with configurable capital, transaction cost rates, and slippage.

## Related Concepts
- OHLCV Data
- Signals and Position Sizing
- Maximum Drawdown
- Sharpe Ratio
