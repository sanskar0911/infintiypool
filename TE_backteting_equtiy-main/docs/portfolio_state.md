# Portfolio State & Position Tracking

QUANTLAB tracks complete portfolio state at every bar:

$$\text{Portfolio Value} = \text{Cash Balance} + \sum (\text{Quantity}_i \times \text{Current Price}_i)$$

## Key Components

1. **Cash Balance**: Available unallocated liquid funds.
2. **Open Positions**: Quantity, weighted average entry price, current mark-to-market price, and market value.
3. **Realized P&L**: Profit/Loss locked in from completed sales (net of transaction costs).
4. **Unrealized P&L**: Paper Profit/Loss on open positions $(\text{Current Price} - \text{Avg Entry Price}) \times \text{Quantity}$.
5. **Exposure**: Fraction of total portfolio value invested in open stock holdings:
   $$\text{Exposure} = \frac{\text{Market Value of Holdings}}{\text{Total Portfolio Value}}$$
