# Relative Strength Index (RSI) Mean Reversion Strategy

## Definition
The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions in price.

## Formula
$$\text{RSI} = 100 - \left( \frac{100}{1 + \text{RS}} \right)$$

Where $\text{RS} = \frac{\text{Average Gain over } N \text{ periods}}{\text{Average Loss over } N \text{ periods}}$ (typically $N = 14$).

## Signals
- **Oversold Buy Signal**: RSI drops below 30 (or custom threshold), indicating potential price rebound.
- **Overbought Sell Signal**: RSI rises above 70 (or custom threshold), indicating potential price pullback.

## Characteristics
RSI performs best in range-bound, mean-reverting market regimes. It struggles during strong persistent bull or bear trends where prices remain overbought or oversold for extended durations.
