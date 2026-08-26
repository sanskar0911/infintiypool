# Transaction Costs and Slippage

## Definition
Transaction costs and execution slippage represent real-world market friction that reduces backtest profitability compared to theoretical price signals.

## Components
1. **Commission / Brokerage**: Fixed or percentage fee per order (e.g. 0.1% or INR 20 per trade).
2. **Slippage**: The difference between the expected signal price and the actual execution price due to market impact and bid-ask spreads (e.g. 0.05%).
3. **Statutory Levies**: Securities Transaction Tax (STT), exchange turnover fees, and GST.

## Backtest Friction Formula
$$\text{Execution Price}_{\text{BUY}} = P_{\text{close}} \times (1 + \text{slippage\_rate})$$

$$\text{Execution Price}_{\text{SELL}} = P_{\text{close}} \times (1 - \text{slippage\_rate})$$
