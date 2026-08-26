# Look-Ahead Bias in Quantitative Backtesting

## Definition
Look-ahead bias occurs when a backtest utilizes information that was not available at the time the trading decision would have been made in live market conditions.

## Common Causes
- Using closing prices $P_t$ to execute trades at $t$ without accounting for execution latency.
- Computing indicators using future price data or post-hoc data adjustments (e.g. corporate action restatements).
- Utilizing global dataset metrics (such as global max price) inside trading loops.

## Prevention Protocols
- Always shift signals by 1 period ($\text{Signal}_{t-1}$) or execute at the open/close of the subsequent bar ($t+1$).
- Enforce strict sequential point-in-time calculation routines in backtesting engines.
