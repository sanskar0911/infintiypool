# Simple Moving Average (SMA) Crossover Strategy

## Definition
The Simple Moving Average (SMA) crossover strategy is a classic trend-following quantitative trading model. It calculates the unweighted arithmetic mean of a stock's closing prices over specified lookback windows.

## Formula
$$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$

Where $P_t$ is the closing price at time $t$, and $k$ is the window period (e.g. 20 days for short window, 50 days for long window).

## Signals
- **Buy Signal (Golden Cross)**: Generated when the short-term SMA crosses above the long-term SMA ($\text{SMA}_{\text{short}} > \text{SMA}_{\text{long}}$).
- **Sell Signal (Death Cross)**: Generated when the short-term SMA crosses below the long-term SMA ($\text{SMA}_{\text{short}} < \text{SMA}_{\text{long}}$).

## Strengths & Weaknesses
- **Strengths**: Captures major sustained multi-month market trends; easy to compute and interpret.
- **Weaknesses**: Prone to whipsaws and false crossovers during sideways or range-bound market regimes; lags behind sudden market reversals due to equal weighting of past prices.
