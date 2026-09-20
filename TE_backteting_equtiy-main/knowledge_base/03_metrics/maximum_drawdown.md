# Maximum Drawdown

## Definition
Maximum Drawdown (MDD) is the maximum observed loss from a peak to a trough of a portfolio before a new peak is achieved, expressed as a percentage.

## Why It Matters
Maximum Drawdown is the single most important risk metric for downside exposure assessment. It quantifies the worst-case capital loss an investor would have experienced holding the strategy.

## Core Concept
Drawdown tracks equity deterioration relative to its historical high-water mark (peak). MDD identifies the deepest equity curve valley across the entire backtest horizon.

## How It Works
1. Track cumulative peak portfolio value over time: $P_t = \max_{0 \le s \le t} (V_s)$.
2. Calculate instantaneous percentage drawdown at step $t$: $DD_t = \frac{V_t - P_t}{P_t}$.
3. Maximum Drawdown is the minimum (most negative) value of $DD_t$.

## Formula
$$\text{Drawdown}_t = \frac{V_t - \text{Peak}_t}{\text{Peak}_t}$$
$$\text{Maximum Drawdown} = \min_{t} \left( \text{Drawdown}_t \right)$$

## Example
If a portfolio starts at ₹100,000, rises to a peak of ₹150,000, drops to ₹105,000, and later recovers to ₹160,000:
$$\text{Drawdown at trough} = \frac{105,000 - 150,000}{150,000} = -30.0\%$$
Thus, Maximum Drawdown = 30.0%.

## Common Mistakes
- Misinterpreting drawdown as loss from initial capital rather than loss from peak.
- Ignoring drawdown duration (the time taken to recover back to peak).

## Backtesting Considerations
To recover from a 50% drawdown, a portfolio must achieve a subsequent +100% gain just to break even.

## QUANTLAB Implementation
`src/metrics.py` calculates Max Drawdown percentage and peak-to-trough series; `src/visualization.py` plots interactive drawdown charts.

## Related Concepts
- Sortino Ratio
- Recovery Factor
- Risk Management & Stop Loss
