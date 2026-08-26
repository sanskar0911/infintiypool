# Sharpe Ratio

## Definition
The Sharpe Ratio measures the risk-adjusted performance of an investment portfolio by comparing its excess return above the risk-free rate to its total volatility.

## Formula
$$\text{Sharpe Ratio} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p} \times \sqrt{252}$$

Where $R_p$ is daily portfolio return, $R_f$ is risk-free rate (assumed 0.0 in basic backtests), and $\sigma_p$ is the standard deviation of daily portfolio returns.

## Industry Benchmarks
- **< 0.50**: Poor risk-adjusted return (unacceptable for institutional allocation).
- **0.50 – 1.00**: Good risk-adjusted return for systematic equity strategies.
- **> 1.00**: Excellent risk-adjusted return (hedge fund quality).
