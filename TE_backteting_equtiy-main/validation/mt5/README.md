# MetaTrader 5 (MT5) Custom Symbol Bar Conversion Guide

This documentation outlines the conversion pipeline from QUANTLAB daily OHLCV datasets to MetaTrader 5 (MT5) custom symbol bar import format.

---

## 1. Why the Conversion is Required

QUANTLAB is a custom Python-based quantitative backtesting engine. When cross-validating backtest results, order executions, and indicator values against MetaTrader 5's Strategy Tester:
- MetaTrader 5 cannot directly ingest arbitrary Python/pandas CSV files containing custom column order, derived indicators, or raw date strings.
- MT5 requires an exact bar structure (`<DateTime>,<Open>,<High>,<Low>,<Close>,<TickVolume>,<RealVolume>,<Spread>`) with strict date-time formatting (`YYYY.MM.DD HH:MM:SS`) to populate its internal historical database (`.hcc` files).
- Ensuring an exact 1:1 match in historical prices between QUANTLAB and MT5 allows engineers to isolate execution logic differences (slippage, fill modeling, spread costs) from data discrepancies.

---

## 2. Original QUANTLAB CSV Structure

The source CSV dataset (e.g., `data/raw/RELIANCE.csv` or `data/processed/RELIANCE.csv`) typically contains:

| Column | Description | Data Type |
| :--- | :--- | :--- |
| `Date` | Calendar trading date (`YYYY-MM-DD`) | String / ISO Date |
| `Open` | Actual unadjusted opening price | Float |
| `High` | Actual unadjusted highest traded price | Float |
| `Low` | Actual unadjusted lowest traded price | Float |
| `Close` | Actual unadjusted closing price | Float |
| `Adj Close` | Adjusted closing price (dividends/splits adjusted) | Float |
| `Volume` | Actual exchange traded share volume | Integer / Float |
| `Returns` | Daily simple or log returns (processed only) | Float |
| `MA50` | 50-day moving average (processed only) | Float |

---

## 3. MT5 Target Structure

The output file (`validation/mt5/RELIANCE_MT5.csv` and `data/metatrader5/RELIANCE_MT5.csv`) strictly adheres to MetaTrader 5 Custom Symbol Bar Import standards:

```csv
<DateTime>,<Open>,<High>,<Low>,<Close>,<TickVolume>,<RealVolume>,<Spread>
2020.01.01 00:00:00,693.9785156,698.1387329,688.2639771,690.1383057,1,14004468,0
2020.01.02 00:00:00,691.2355347,704.47052,691.2355347,701.8875122,1,17710316,0
```

---

## 4. Meaning of Each MT5 Field

1. `<DateTime>`:
   Combined date and time timestamp in `YYYY.MM.DD HH:MM:SS` format.
2. `<Open>`:
   The opening traded price of the bar.
3. `<High>`:
   The maximum traded price reached during the bar interval.
4. `<Low>`:
   The minimum traded price reached during the bar interval.
5. `<Close>`:
   The final settlement / closing traded price of the bar.
6. `<TickVolume>`:
   The count of price revisions (ticks) recorded during the bar.
7. `<RealVolume>`:
   The total number of traded contracts or shares executed during the bar.
8. `<Spread>`:
   The bid-ask spread at bar closing in points (or 0 for fixed spread modeling).

---

## 5. Why Adjusted Close is Excluded

- **Price Integrity & Execution Realism**: Adjusted Close recalculates historical prices retroactively to account for corporate actions (stock splits, bonus issues, dividends).
- **Misleading OHLC Boundaries**: If Adjusted Close is substituted for Close, the relationship between Open, High, Low, and Close is distorted (e.g., Close may appear outside the High-Low range of historical trading).
- **MT5 Strategy Tester Expectation**: MT5 executes orders against actual market price levels (Open, High, Low, Close). Using adjusted prices would distort price triggers, limit orders, and stop-loss levels. Corporate actions should be modeled as cash flow adjustments rather than price shifts.

---

## 6. Why Returns and MA50 are Excluded

- **Derived Indicators**: `Returns` and `MA50` are downstream computed metrics, not primary market feed attributes.
- **MT5 Architecture**: MT5 calculates all technical indicators (Moving Averages, RSI, ATR) dynamically on top of the imported OHLC bars using MQL5 code or native indicator buffers. Including auxiliary columns would cause MT5 bar import parsers to fail.

---

## 7. How TickVolume is Handled

- **The Problem**: Daily equity data from standard historical feeds provides total daily traded volume (`RealVolume`), but does not track the number of distinct intraday price changes (`TickVolume`).
- **Deterministic Rule**:
  - **`constant` mode (Default)**: Sets `<TickVolume>` to `1` for every valid daily bar. This ensures MT5 bar validity without fabricating artificial tick dynamics.
  - **`volume` mode**: Maps the source `Volume` value directly to `<TickVolume>`. Note: This is an explicit compatibility placeholder for indicators that specifically require non-zero tick volume, and does not represent actual market tick counts.

---

## 8. How RealVolume is Mapped

- The source dataset's exchange `Volume` field (total number of shares traded on that day) is preserved and mapped directly to `<RealVolume>` as an integer.
- No scaling, clipping, or rounding of volume occurs.

---

## 9. How Spread is Handled

- MT5 requires an integer value for `<Spread>` in points.
- By default, `<Spread>` is set to `0` (configurable via `--spread <val>`). In MT5 Strategy Tester, spread can be either taken from the bar data or configured directly in the Symbol specifications or Tester settings.

---

## 10. Why Daily Bars Use `00:00:00`

- The source dataset consists of daily end-of-day (EOD) records without intraday timestamps.
- MetaTrader 5 requires a complete timestamp with time components (`HH:MM:SS`).
- In accordance with standard MT5 daily bar conventions, `00:00:00` is the standard anchor timestamp for D1 (daily) bars.
- No intraday bars or synthetic minute candles are created or interpolated.

---

## 11. How to Import the Generated File into MT5 Custom Symbol

Follow these steps in the MetaTrader 5 Terminal:

1. **Open Symbols Window**:
   - In MT5, press `Ctrl + U` (or go to **View** → **Symbols**).
2. **Create a Custom Symbol**:
   - Select the **Custom** category in the left tree.
   - Click **Create Custom Symbol**.
   - Set:
     - **Symbol**: e.g., `RELIANCE_CUSTOM`
     - **Path**: `Custom\Equities`
     - **Digits**: `2` (or precision corresponding to your asset)
     - **Contract size**: `1`
     - **Calculation**: `Exchange Stocks`
   - Click **OK**.
3. **Navigate to Bars Tab**:
   - In the Symbols window, select your newly created custom symbol.
   - Click on the **Bars** tab.
   - Set the timeframe selector to **D1** (Daily).
4. **Import the Converted CSV**:
   - Click **Import Bars**.
   - Browse and select `RELIANCE_MT5.csv` (from `validation/mt5/` or `data/metatrader5/`).
   - MT5 will parse the headers `<DateTime>,<Open>,<High>,<Low>,<Close>,<TickVolume>,<RealVolume>,<Spread>` and preview the rows.
   - Click **OK** to commit the bars into the MT5 database.
5. **Add to Market Watch**:
   - Open **Market Watch** (`Ctrl + M`), right-click, select **Symbols**, and enable your custom symbol.
   - Open a daily chart to visually confirm the candlestick history.

---

## 12. Limitations of Converting Daily OHLCV into MT5 Bars

1. **Lack of Intraday Tick Dynamics**:
   - A single daily bar does not capture the intraday sequence of price movement (whether the High was reached before the Low, or vice-versa).
   - In MT5 Strategy Tester, running "Every tick based on real ticks" is not possible on daily bar imports without high-resolution tick data. Testing should use "Open prices only" or standard bar-model simulations.
2. **Holiday / Weekend Gaps**:
   - The converter preserves exact trading dates and does not synthesize candles for holidays or weekends. MT5 handles calendar gaps natively.
3. **Market Session Timings**:
   - The bar timestamp is normalized to `00:00:00` rather than the exact market exchange open time (e.g., `09:15:00` IST). This is standard for daily MT5 bars.

---

## 13. Running the Converter Tool

### Basic Conversion
```bash
python tools/convert_to_mt5.py \
    --input data/raw/RELIANCE.csv \
    --output validation/mt5/RELIANCE_MT5.csv
```

### Preview First 10 Converted Bars
```bash
python tools/convert_to_mt5.py \
    --input data/raw/RELIANCE.csv \
    --output validation/mt5/RELIANCE_MT5.csv \
    --preview 10
```

### Custom Tick Volume & Spread Options
```bash
python tools/convert_to_mt5.py \
    --input data/raw/RELIANCE.csv \
    --output validation/mt5/RELIANCE_MT5.csv \
    --tick-volume-mode constant \
    --spread 0
```
