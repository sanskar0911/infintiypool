# Sharpe Ratio

## Definition
The Sharpe Ratio is a fundamental quantitative metric that measures the risk-adjusted return of an investment portfolio or trading strategy relative to a risk-free rate.

## Why It Matters
A high raw return is meaningless if achieved through extreme volatility or dangerous drawdowns. The Sharpe Ratio allows quants to compare strategies on a normalized risk-adjusted basis.

## Core Concept
The ratio quantifies how much excess return a strategy yields per unit of total risk (standard deviation of returns). A higher Sharpe ratio indicates superior risk efficiency.

## How It Works
1. Calculate daily return series $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$.
2. Subtract annualized risk-free rate $R_f$ (e.g., 5% government bond yield).
3. Compute mean excess return $\bar{R}_{\text{excess}} = \text{Mean}(R_t) - R_f / 252$.
4. Divide mean excess return by the standard deviation of daily returns $\sigma_{R}$.
5. Annualize by multiplying by $\sqrt{252}$.

## Formula
$$\text{Sharpe Ratio} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p} \times \sqrt{252}$$

Where:
- $R_p$ = Daily Portfolio Returns
- $R_f$ = Daily Risk-Free Rate
- $\sigma_p$ = Standard Deviation of Portfolio Returns
- $252$ = Annual trading days

## Example
If a strategy achieves an annualized return of 18% with an annual volatility of 10% and risk-free rate is 5%:
$$\text{Sharpe Ratio} = \frac{0.18 - 0.05}{0.10} = 1.30$$

## Common Mistakes
- Annualizing using calendar days (365) instead of trading days (252).
- Ignoring risk-free rates when evaluating equity strategies.
- Assuming return distribution is strictly normal (ignoring skewness and fat tails).

## Backtesting Considerations
Strategies with high Sharpe ratios (>2.0) in backtests often suffer from overfitting or unmodeled transaction slippage.

## QUANTLAB Implementation
In QUANTLAB, `src/metrics.py` calculates the annualized Sharpe Ratio using daily portfolio valuation series.

## Related Concepts
- Sortino Ratio
- Annual Volatility
- CAGR
- Maximum Drawdown
