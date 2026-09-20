# Relative Strength Index (RSI) Mean-Reversion Strategy

## Definition
The Relative Strength Index (RSI) strategy is a momentum oscillator strategy that measures the speed and change of price movements on a scale from 0 to 100 to identify oversold and overbought market conditions.

## Why It Matters
RSI provides quants with a mean-reversion tool to capitalize on short-term price overextensions in range-bound markets.

## Core Concept
- **Oversold (<30)**: Price has fallen sharply; expect mean-reversion rebound $\rightarrow$ BUY (+1).
- **Overbought (>70)**: Price has risen sharply; expect mean-reversion pullback $\rightarrow$ SELL (-1).

## How It Works
1. Calculate daily gain $U$ and daily loss $D$.
2. Compute Exponential Moving Average of gains $RS = \frac{\text{EMA}(U, 14)}{\text{EMA}(D, 14)}$.
3. Compute $\text{RSI} = 100 - \left( \frac{100}{1 + RS} \right)$.

## Formula
$$\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}$$

## Example
If 14-day RSI for TCS drops to 26.5 (below 30 oversold threshold), a BUY signal is generated.

## Common Mistakes
- Buying oversold stocks during a strong structural downtrend.
- Selling overbought stocks during an aggressive bull trend.

## Backtesting Considerations
Best combined with trend-filtering indicators (e.g. SMA 200) to avoid trading against major market trends.

## QUANTLAB Implementation
Implemented in `src/strategy.py` (`generate_signals(df, strategy_name='RSI', period=14, oversold=30.0, overbought=70.0)`).

## Related Concepts
- Mean Reversion
- Momentum Oscillators
- Overbought / Oversold
