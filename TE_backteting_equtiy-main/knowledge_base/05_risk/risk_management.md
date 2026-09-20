# Quantitative Risk Management

## Definition
Quantitative risk management is the systematic practice of measuring, monitoring, and controlling market risk exposures, drawdowns, and leverage in algorithmic trading.

## Why It Matters
Even high-performing strategies will experience unexpected market shocks or structural regime shifts. Strict risk management prevents total portfolio ruin.

## Core Concept
Key risk management pillars include:
- Position Sizing (e.g. fixed fractional capital allocation per trade).
- Stop Loss & Take Profit limits.
- Portfolio Diversification & Sector Exposure limits.

## How It Works
1. Calculate volatility and maximum risk allowance per trade.
2. Determine position size based on current portfolio value and stop distance.
3. Liquidate positions if stop loss threshold is breached.

## Formula
$$\text{Position Size (Shares)} = \frac{\text{Capital} \times \text{Risk \%}}{\text{Entry Price} - \text{Stop Loss Price}}$$

## Example
Allocating a maximum of 20% of capital (₹20,000) to a single stock with a 5% stop loss limit.

## Common Mistakes
- Over-leveraging positions without stop loss boundaries.
- Increasing position size after consecutive losses (Martingale bias).

## Backtesting Considerations
Always evaluate strategy performance with and without stop loss/take profit mechanisms.

## QUANTLAB Implementation
Configurable position sizing, stop loss, and take profit parameters in `src/backtester.py`.

## Related Concepts
- Maximum Drawdown
- Stop Loss
- Position Sizing
