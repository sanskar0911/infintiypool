# OHLCV Data Standard

## Definition
OHLCV stands for Open, High, Low, Close, and Volume. It represents the standard tabular bar data format for representing financial price action over discrete time intervals (e.g., 1-day, 1-hour).

## Why It Matters
OHLCV bars are the foundational input for technical analysis, indicator calculation (SMA, EMA, RSI), and price simulation in quantitative backtesting engines.

## Core Concept
Each row in an OHLCV dataset records:
- **Open**: The initial trading price at bar start.
- **High**: The maximum price reached during the bar interval.
- **Low**: The minimum price reached during the bar interval.
- **Close**: The final trading price at bar end.
- **Volume**: Total quantity of shares or contracts traded during the bar.

## How It Works
1. Data feed provides raw trade prints.
2. Trades are aggregated into regular time buckets.
3. High and Low bounds are determined alongside bar start Open and end Close.
4. Bar data is cleaned for split adjustments and missing values.

## Formula
$$\text{Bar Range} = \text{High}_t - \text{Low}_t$$
$$\text{Bar Return} = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$

## Example
For INFY on 2024-01-15: Open = ₹1500.0, High = ₹1525.0, Low = ₹1490.0, Close = ₹1520.0, Volume = 2,500,000.

## Common Mistakes
- Using unadjusted Close prices for stocks with historical split or dividend events.
- Forward filling missing bar prices without adjusting volume.

## Backtesting Considerations
Execution prices should consider slippage within the High-Low range of the execution bar.

## QUANTLAB Implementation
QUANTLAB ingests processed CSV data files in `data/processed/{TICKER}.csv` with pandas datetime index and standard `Close` columns.

## Related Concepts
- Historical Data
- Execution Friction
- Signals
