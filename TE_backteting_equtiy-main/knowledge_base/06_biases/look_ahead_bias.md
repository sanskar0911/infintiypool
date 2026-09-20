# Look-Ahead Bias

## Definition
Look-Ahead Bias is a critical flaw in quantitative backtesting where data that was unavailable at the simulated point in time is inadvertently utilized to generate trading signals or calculate positions.

## Why It Matters
Look-ahead bias corrupts backtest validity by granting the algorithm "future vision," producing artificially stellar performance metrics that inevitably collapse in live trading.

## Core Concept
In live trading, decisions at time $t$ can only use information up to time $t$. If a backtest uses Close price at time $t$ to execute at Open price of time $t$, or uses fundamental data before public release, it commits look-ahead bias.

## How It Works
1. Strategy evaluates signal using $P_t$.
2. Backtester fills trade at $P_{t-1}$ or uses $P_t$ high/low ranges to predict entry before they occurred.
3. Performance statistics skyrocket unnaturally.

## Formula
$$\text{Signal}_t = f(\{X_0, X_1, \dots, X_t\}) \quad \text{(Valid Point-in-Time)}$$
$$\text{Signal}_t = f(\{X_0, X_1, \dots, X_{t+1}\}) \quad \text{(Look-Ahead Violation)}$$

## Example
Calculating 20-day SMA using the *current bar's Close price* to execute a trade at the *current bar's Open price*.

## Common Mistakes
- Filling trades at the current bar's Open using signals derived from the current bar's Close.
- Using unadjusted financial statements prior to official filing dates.

## Backtesting Considerations
Shift signals by 1 bar (`df['Signal'].shift(1)`) or execute orders on the Open price of bar $t+1$.

## QUANTLAB Implementation
In QUANTLAB, `src/backtester.py` enforces explicit bar lag on signal transitions to preserve point-in-time correctness.

## Related Concepts
- Point-in-time Correctness
- Survivorship Bias
- Overfitting
