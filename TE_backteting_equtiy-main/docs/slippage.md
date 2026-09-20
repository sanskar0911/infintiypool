# Slippage Model

QUANTLAB applies adverse execution price slippage to model market impact and bid-ask spread friction.

## Slippage Calculation

Slippage is specified in basis points (`slippage_bps`), where $10\text{ bps} = 0.10\%$.

- **BUY Fill Price**:
  $$\text{Fill Price} = \text{Base Price} \times \left(1 + \frac{\text{slippage\_bps}}{10000}\right)$$
- **SELL Fill Price**:
  $$\text{Fill Price} = \text{Base Price} \times \left(1 - \frac{\text{slippage\_bps}}{10000}\right)$$

This guarantees that trade executions occur at prices less favorable to the trader, matching real-world execution dynamics.
