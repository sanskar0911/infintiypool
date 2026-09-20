# Transaction Costs and Slippage

## Definition
Transaction costs and slippage represent execution friction in real trading. Transaction costs include broker commissions, exchange transaction fees, stamp duty, and taxes. Slippage represents the difference between expected signal execution price and actual fill price.

## Why It Matters
Ignoring market friction causes backtests to produce wildly unrealistic phantom profits. High-frequency or high-turnover strategies can quickly become unprofitable when execution fees and spread slippage are subtracted.

## Core Concept
- **Brokerage Commission Rate**: Fixed percentage fee per trade execution (e.g., 0.1% or 0.001).
- **Slippage Rate**: Execution penalty due to market impact and bid-ask spread (e.g., 0.05% or 0.0005).

## How It Works
1. When BUY signal occurs at price $P$, actual fill price is $P_{\text{buy}} = P \times (1 + \text{slippage\_rate})$.
2. Transaction fee charged = $P_{\text{buy}} \times \text{Shares} \times \text{commission\_rate}$.
3. When SELL signal occurs at price $P$, actual fill price is $P_{\text{sell}} = P \times (1 - \text{slippage\_rate})$.

## Formula
$$\text{Cost}_{\text{trade}} = (\text{Shares} \times P_{\text{exec}}) \times (\text{Commission Rate} + \text{Slippage Rate})$$

## Example
Buying 1,000 shares at ₹1,000 with 0.1% commission and 0.05% slippage results in an execution fill at ₹1,000.50 and total commission of ₹1,000.50.

## Common Mistakes
- Setting commission to 0.0% during backtesting.
- Assuming infinite liquidity at the Close price of the signal bar.

## Backtesting Considerations
Model separate commission and slippage rates for realistic stress testing.

## QUANTLAB Implementation
In QUANTLAB, `src/backtester.py` accepts `txn_cost_rate` and `slippage_rate` arguments and subtracts execution costs dynamically from cash reserves.

## Related Concepts
- Order Execution
- Slippage
- Bid-Ask Spread
