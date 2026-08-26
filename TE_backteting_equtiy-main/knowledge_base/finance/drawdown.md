# Maximum Drawdown

## Definition
Maximum Drawdown (MDD) measures the largest percentage peak-to-trough decline in portfolio value before a new peak is attained. It quantifies worst-case capital loss risk.

## Formula
$$\text{Drawdown}_t = \frac{V_t - \max_{\tau \le t}(V_\tau)}{\max_{\tau \le t}(V_\tau)}$$

$$\text{Max Drawdown} = \min_{t} (\text{Drawdown}_t)$$

Where $V_t$ is the portfolio value at time $t$.

## Recovery Dynamics
To recover from a drawdown of $D$, the portfolio must subsequently gain:
$$\text{Required Recovery Gain} = \frac{D}{1 - D}$$

For example, a 20% drawdown requires a 25% gain to reach breakeven, while a 50% drawdown requires a 100% gain.
