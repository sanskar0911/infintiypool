# Exponential Moving Average (EMA) Crossover Strategy

## Definition
The Exponential Moving Average (EMA) is a momentum indicator that places a greater weight and significance on the most recent price data points, making it more responsive to fresh price action than the Simple Moving Average.

## Formula
$$\text{EMA}_t = (P_t \times \alpha) + (\text{EMA}_{t-1} \times (1 - \alpha))$$

Where multiplier $\alpha = \frac{2}{k + 1}$, and $k$ is the smoothing period.

## Signals
- **Buy Signal**: Occurs when short-term EMA (e.g. 12-day) crosses above long-term EMA (e.g. 26-day).
- **Sell Signal**: Occurs when short-term EMA crosses below long-term EMA.

## Market Behavior
EMA responds faster to trend changes than SMA, making it suitable for volatile or fast-trending equity markets. However, its higher sensitivity increases the risk of false signals during short-term market noise.
