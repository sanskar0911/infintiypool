# Transaction Cost Model

QUANTLAB models brokerage commissions dynamically for every trade fill.

## Commission Calculation

Commission is specified in basis points (`commission_bps`), where $10\text{ bps} = 0.10\%$.

$$\text{Gross Value} = \text{Quantity} \times \text{Fill Price}$$
$$\text{Commission} = \text{Gross Value} \times \left( \frac{\text{commission\_bps}}{10000} \right)$$

- **BUY Order**: Cash outflow = $\text{Gross Value} + \text{Commission}$
- **SELL Order**: Cash inflow = $\text{Gross Value} - \text{Commission}$
