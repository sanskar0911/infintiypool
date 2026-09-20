# Simple Moving Average (SMA) Crossover Strategy

## Definition
The Simple Moving Average (SMA) Crossover is a trend-following quantitative equity strategy that generates buy/sell signals when a short-term moving average crosses above or below a long-term moving average.

## Why It Matters
SMA crossover is one of the most widely used momentum strategies in quantitative finance due to its mathematical simplicity, trend identification capabilities, and lack of parameter complexity.

## Core Concept
- **Golden Cross (Bullish Signal)**: Fast SMA crosses above Slow SMA $\rightarrow$ Buy (+1).
- **Death Cross (Bearish Signal)**: Fast SMA crosses below Slow SMA $\rightarrow$ Sell (-1) or Cash (0).

## How It Works
1. Compute short SMA (e.g. 20-day) and long SMA (e.g. 50-day) over Close prices.
2. Signal $+1$ is triggered when $\text{SMA}_{20} > \text{SMA}_{50}$.
3. Signal $-1$ is triggered when $\text{SMA}_{20} < \text{SMA}_{50}$.

## Formula
$$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$

## Example
If 20-day SMA of INFY rises to ₹1520 and crosses above 50-day SMA at ₹1510, a LONG signal (+1) is generated.

## Common Mistakes
- Whipsaw losses during range-bound (sideways) market regimes.
- Parameter curve-fitting (e.g. optimizing windows to historical noise).

## Backtesting Considerations
Trend-following strategies suffer lag; slippage and transaction costs must be accounted for on signal changes.

## QUANTLAB Implementation
Implemented in `src/strategy.py` (`generate_signals(df, strategy_name='SMA', short_window=20, long_window=50)`).

## Related Concepts
- EMA Crossover
- Moving Averages
- Trend Following
