# Execution Model & Timing

## Look-Ahead Bias Prevention

To eliminate look-ahead bias, QUANTLAB enforces point-in-time execution timing. When a trading signal is triggered at the Close of bar $T$, the execution price cannot assume instant execution at the same Close price.

## Supported Timing Modes

1. **`NEXT_BAR_OPEN` (Default & Recommended)**
   - Signal generated at bar $T$ Close.
   - Order generated at bar $T$.
   - Executed at bar $T+1$ Open price.
   - Base Execution Price = Open price of bar $T+1$.

2. **`SAME_BAR_CLOSE`**
   - Signal generated at bar $T$.
   - Executed at bar $T$ Close price.
   - Retained for legacy comparison.

3. **`NEXT_BAR_CLOSE`**
   - Signal generated at bar $T$.
   - Executed at bar $T+1$ Close price.
