# Indian Equity Quantitative Backtesting Engine

An institutional-grade, multi-stock, event-driven backtesting engine and interactive analytics dashboard built specifically for Indian cash equity markets (NSE/BSE).

---

## Table of Contents
1. [Engine Architecture & Execution Flow](#1-engine-architecture--execution-flow)
2. [Signal Ingestion & Data Preparation](#2-signal-ingestion--data-preparation)
3. [Position Sizing & Cash Management](#3-position-sizing--cash-management)
4. [Order Execution, Slippage & Transaction Costs](#4-order-execution-slippage--transaction-costs)
5. [Trade Lifecycle & How Trades Are Closed](#5-trade-lifecycle--how-trades-are-closed)
6. [Why & How Trades Are Skipped or Rejected](#6-why--how-trades-are-skipped-or-rejected)
7. [Comprehensive Metrics Calculation & Formulas](#7-comprehensive-metrics-calculation--formulas)
8. [Web Dashboard & API](#8-web-dashboard--api)
9. [Peer Verification & Reproducibility Checklist](#9-peer-verification--reproducibility-checklist)

---

## 1. Engine Architecture & Execution Flow

The engine operates on a strict, chronological event-driven architecture designed to eliminate lookahead bias and survivorship discrepancies.

```
[Signal File (.csv)] ──► [Signal Ingestion Layer] ──► [Timeline Alignment]
                                                             │
┌────────────────────────────────────────────────────────────┴────────────────────────────────┐
│ Daily Bar Event Loop (Chronological Order across all dates t = 1 ... N)                     │
│                                                                                             │
│  Step 1: Read Market Data (Open, High, Low, Close) for bar t                                │
│  Step 2: Determine Actionable Signals (Signals generated on bar t executed on bar t)        │
│  Step 3: Execute SELLS First (Releases cash from exits back into Available Cash)            │
│  Step 4: Execute BUYS Second (Sizes integer shares using current Available Cash)            │
│  Step 5: Mark-to-Market (MTM) Portfolio Valuation at bar t Close                           │
│  Step 6: Update Equity Curve, High-Water Mark & Portfolio Invariants                        │
└────────────────────────────────────────────────────────────┬────────────────────────────────┘
                                                             ▼
                                      [Terminal Force-Close at Final Bar]
                                                             ▼
                                          [Metrics & Analytics Layer]
                                                             ▼
                                     [Web UI / Trade Ledger / CSV Export]
```

---

## 2. Signal Ingestion & Data Preparation

The engine dynamically supports both standard multi-stock signal files and trade-pair signal files (such as `signalsmain2.py.csv`):

### Supported Signal Formats:
1. **Trade-Pair Format** (e.g. `signalsmain2.py.csv`):
   * Columns: `Stock`, `Entry Date`, `Entry Price`, `Exit Date`, `Close Price`
   * Entry rows with valid dates generate `BUY` events (`Signal = +1`).
   * Exit rows with valid dates generate `SELL` events (`Signal = -1`).
   * If `Exit Date` is marked as `"Active(on Jul 31, 2024)"` or similar text, the date fails parsing into a timestamp and is coerced to `NaT` (Not-a-Time). **No exit event is emitted**, keeping the position open and active in the portfolio until either an explicit exit or terminal close.
2. **Timeline 3-Column Format**:
   * Columns: `Company`, `Date`, `Signal` (+1 for Buy, -1 for Sell).

### Data Synchronization:
* The engine aligns all company signals against historical daily OHLCV price series (from the NSE500 database).
* If a stock's historical CSV is absent from local storage, the engine synthesizes its active trading timeline directly from the signal entry and exit prices so the backtest executes with 100% data coverage without crashing.

---

## 3. Position Sizing & Cash Management

* **Baseline Capital**: ₹1,00,00,000 (₹1.00 Crore INR).
* **Sizing Policy**: **Fixed Fraction of Available Cash (`fixed_fraction`)**.
* **Fraction per Trade**: **5.0% (`0.05`)**.

### The Sizing Algorithm (`compute_quantity`):
1. **Cash Allocation**:
   $$\text{Cash Allocation} = \text{Available Cash} \times 0.05$$
2. **Pre-Deduction of Transaction Costs (Affordability Principle)**:
   Purchasing shares incurs brokerage, stamp duty, and GST. If the engine allocated exactly ₹5,00,000 worth of gross shares, taxes would push the total debit above ₹5,00,000, eventually causing cash overdrafts.
   Therefore, the engine calculates the **maximum gross trade value** that can be purchased such that:
   $$\text{Gross Trade Value} + \text{Buy-Side Costs}(\text{Gross Trade Value}) \le \text{Cash Allocation}$$
   $$\text{Max Affordable Value} = \frac{\text{Cash Allocation}}{1 + \text{Buy-Side Cost Rate}}$$
   *(where $\text{Buy-Side Cost Rate} = 0.0504\%$)*.
3. **Whole-Share Truncation (No Fractional Shares)**:
   $$\text{Quantity} = \left\lfloor \frac{\text{Max Affordable Value}}{\text{Execution Price}} \right\rfloor$$
   If $\text{Quantity} < 1$, the trade cannot be afforded and is rejected.

---

## 4. Order Execution, Slippage & Transaction Costs

### A. Execution Timing (Same-Day Execution)
* **Model**: **Same-Bar / Same-Day Execution (`same_bar_close`)**.
* **Rule**: When a signal is generated on Date / Bar $T$, the trade **must be executed on the same day when the signal is generated**, rather than deferred to the next day.
* Execution price is determined by the bar's quote with adverse slippage applied.

### B. Slippage Model
* **Model**: Symmetrical percentage slippage (`fixed_pct`) of **0.05% (5 bps)**.
  * **Buy Fill Price**: $\text{Market Price} \times (1 + 0.0005)$
  * **Sell Fill Price**: $\text{Market Price} \times (1 - 0.0005)$

### C. Statutory Indian Equity Delivery Costs & Taxes
The cost model reflects statutory Indian cash equity delivery rules:

| Cost Component | Rate | Applicable Side | Base / Formula |
| :--- | :--- | :--- | :--- |
| **Brokerage** | **0.03% (0.0003)** | Buy & Sell | $\text{Trade Value} \times 0.0003$ |
| **STT (Securities Transaction Tax)** | **0.10% (0.0010)** | **Sell Side Only** | $\text{Trade Value} \times 0.0010$ (Delivery equity is exempt on Buy side) |
| **Stamp Duty** | **0.015% (0.00015)** | **Buy Side Only** | $\text{Trade Value} \times 0.00015$ (Transfer tax on buy side only) |
| **GST (Goods & Services Tax)** | **18.0% (0.18)** | Buy & Sell | $18\% \times \text{Brokerage}$ (GST applies to brokerage only) |
| **Exchange Charges** | **0.0%** | Baseline | 0.0% |

#### Net Cost Summary:
* **Total Buy-Side Cost**:
  $$\text{Buy Cost Rate} = 0.0003 + 0.00015 + (0.0003 \times 0.18) = \mathbf{0.0504\%}$$
* **Total Sell-Side Cost**:
  $$\text{Sell Cost Rate} = 0.0003 + 0.0010 + (0.0003 \times 0.18) = \mathbf{0.1354\%}$$
* **Total Round-Trip Statutory Cost**: $0.0504\% + 0.1354\% = \mathbf{0.1858\%}$.
* **Total Round-Trip Friction (with Slippage)**: $0.1858\% + 0.10\% = \mathbf{0.2858\%}$.

---

## 5. Trade Lifecycle & How Trades Are Closed

A trade undergoes the following state transitions:

```
[Signal File] ──► [Candidate Buy] ──► [Validation & Affordability] ──► [OPEN Position]
                                                                            │
                       ┌────────────────────────────────────────────────────┴────────────────┐
                       ▼                                                                     ▼
               [Normal Exit]                                                       [Terminal Force-Close]
      Trigger: Signal == -1 or Exit Date in CSV                                Trigger: Reached final bar of backtest
      Price: Same-Day Close with Slippage                                      Price: Final Bar Close with Slippage
      Reason: "External SELL Signal"                                          Reason: "End of Data"
```

### 1. Normal Trade Exit
* Occurs when an external `SELL` signal (`Signal == -1`) or a parsed `Exit Date` arrives.
* The order is executed on the same day when signal is generated (minus 0.05% slippage).
* Cash proceeds ($(\text{Quantity} \times \text{Execution Price}) - \text{Sell Costs}$) are immediately credited to the portfolio cash balance.
* Trade record is logged to the ledger with entry date, exit date, gross P&L, transaction costs, and net realized P&L.

### 2. How Open/Active Trades Are Closed at End of Signal File (`End of Data`)
* In any backtest, trades may still be open when the data ends (e.g. trades initiated in the final month or marked `"Active(on Jul 31, 2024)"`).
* **Terminal Rule**: On the very last date of the unified timeline, the engine checks all remaining positions where $\text{Quantity} > 0$.
* Each open position is automatically exited at the **Close price of the final bar** (minus 0.05% slippage).
* The exit reason in the trade log is recorded as **`End of Data`**.
* **Why this is done**:
  * If open positions were simply ignored, invested capital would be "locked" and omitted from realized P&L, producing incorrect final equity and corrupted win rate/profit factor metrics.
  * Force-closing at the final close liquidates all holdings into cash, ensuring:
    $$\text{Final Equity} = \text{Final Cash Balance}$$
    $$\text{Unrealized P&L} = ₹0.00$$
    Every single rupee in the portfolio is cleanly accounted for.

---

## 6. Why & How Trades Are Skipped or Rejected

If a signal in the CSV does not result in an executed trade, the engine records the exact timestamp, symbol, and reason in the `rejected_orders` ledger. Below are the 4 causes:

### 1. Duplicate Position Rule (Already Holding the Stock)
* **Condition**: A `BUY` signal arrives for stock $S$, but the portfolio already has $\text{Quantity} > 0$ in stock $S$.
* **Behavior**: Skipped automatically.
* **Rationale**: The engine operates under a single-position-per-symbol policy (no pyramiding / averaging down unless explicitly configured). The signal is discarded until the active trade exits.

### 2. Insufficient Cash for 1 Whole Share
* **Condition**: $\text{Quantity} = \left\lfloor \frac{\text{Max Affordable Value}}{\text{Execution Price}} \right\rfloor = 0$.
* **Reason in Ledger**: `"insufficient cash for even 1 share (cash=..., price=...)"`.
* **Behavior**: Rejected.
* **Rationale**:
  * If a high-priced stock triggers a buy signal when available cash (or 5% cash allocation) cannot purchase 1 share after taxes, purchasing even 1 share would violate the cash budget and cause negative cash balances.
  * Since the Indian stock market does not allow fractional delivery shares, the order cannot execute.

### 3. Missing Price Data / Inactive Market Bar
* **Condition**: A signal exists for symbol $S$ on date $T$, but the stock has no trading quote, zero volume, or `execution_price <= 0`.
* **Reason in Ledger**: `"invalid market price"`.
* **Behavior**: Rejected.

### 4. Portfolio Level Constraints
* **Condition**: A buy order would breach configured limits (e.g., maximum position weight or maximum total equity exposure).
* **Behavior**: Rejected with the specific constraint failure message.

---

## 7. Comprehensive Metrics Calculation & Formulas

All performance metrics follow standard quantitative finance methodologies.

### Base Parameters:
* **Annual Trading Days**: **252 days**
* **Risk-Free Rate ($R_f$)**: **6.0% per annum (`0.06`)**
* **Daily Risk-Free Rate**: $R_{f,\text{daily}} = \frac{0.06}{252} \approx 0.000238095$

---

### Formulas for the Top 20 Performance Metrics:

#### 1. Total Return (%)
$$\text{Total Return (\%)} = \left(\frac{\text{Final Equity} - \text{Initial Capital}}{\text{Initial Capital}}\right) \times 100$$

#### 2. CAGR (Compound Annual Growth Rate)
Measures the constant annual compounding rate that yields the final equity over the total elapsed trading days $N$:
$$\text{CAGR} = \left(\frac{\text{Final Equity}}{\text{Initial Capital}}\right)^{\frac{252}{N}} - 1$$

#### 3. Total P&L
$$\text{Total P\&L} = \text{Final Equity} - \text{Initial Capital}$$

#### 4. Annualized Volatility ($\sigma_{\text{ann}}$)
Based on daily percentage portfolio returns $r_t = \frac{\text{Portfolio Value}_t - \text{Portfolio Value}_{t-1}}{\text{Portfolio Value}_{t-1}}$:
$$\sigma_{\text{ann}} = \text{Standard Deviation}(r_t) \times \sqrt{252}$$

#### 5. Sharpe Ratio
Measures risk-adjusted excess return over the 6% annual risk-free hurdle rate:
$$\text{Sharpe Ratio} = \frac{\text{Mean}(r_t - R_{f,\text{daily}})}{\text{Std}(r_t)} \times \sqrt{252}$$

#### 6. Sortino Ratio
Measures return against downside risk only (volatility of negative excess returns):
$$\text{Downside Deviation} = \sqrt{\frac{1}{N}\sum_{t=1}^{N} \min\left(0, r_t - R_{f,\text{daily}}\right)^2} \times \sqrt{252}$$
$$\text{Sortino Ratio} = \frac{\text{Mean}(r_t - R_{f,\text{daily}}) \times 252}{\text{Downside Deviation}}$$

#### 7. Calmar Ratio
Ratio of annualized compound return to maximum peak-to-trough drawdown:
$$\text{Calmar Ratio} = \frac{\text{CAGR}}{|\text{Maximum Drawdown Fraction}|}$$

#### 8. Maximum Drawdown (%) & Drawdown in INR
On each day $t$, the Peak (High Watermark) is tracked: $\text{Peak}_t = \max(\text{Peak}_{t-1}, \text{Portfolio Value}_t)$.
$$\text{Drawdown}_t = \frac{\text{Portfolio Value}_t - \text{Peak}_t}{\text{Peak}_t}$$
$$\text{Max Drawdown (\%)} = \min_{t}(\text{Drawdown}_t) \times 100$$
$$\text{Max Drawdown (INR)} = \max_{t}(\text{Peak}_t - \text{Portfolio Value}_t)$$

#### 9. Max Drawdown Duration (days)
The longest contiguous stretch of calendar/trading days during which portfolio equity remained below a previous peak.

#### 10. Win Rate (%)
$$\text{Win Rate} = \left(\frac{\text{Count of Trades with Net P\&L} > 0}{\text{Total Completed Trades}}\right) \times 100$$

#### 11. Profit Factor
$$\text{Profit Factor} = \frac{\sum \text{Net Profit of Winning Trades}}{\sum |\text{Net Loss of Losing Trades}|}$$

#### 12. Risk / Reward Ratio
$$\text{Risk / Reward} = \frac{\text{Average Net Profit per Winning Trade}}{|\text{Average Net Loss per Losing Trade}|}$$

#### 13. Total Trades
Total number of completed buy-and-sell round trips recorded in the ledger.

#### 14. Average P&L per Trade
$$\text{Average P\&L} = \frac{\text{Total Net Realized P\&L}}{\text{Total Trades}}$$

#### 15. Average Win & Average Loss
* **Average Win**: Total net profit of winning trades divided by number of winning trades.
* **Average Loss**: Total net loss of losing trades divided by number of losing trades.

#### 16. Best Trade & Worst Trade
* **Best Trade**: Maximum net realized P&L from a single trade.
* **Worst Trade**: Minimum (most negative) net realized P&L from a single trade.

#### 17. Average Holding Period (days)
$$\text{Avg Holding Period} = \frac{1}{M}\sum_{k=1}^{M} (\text{Exit Date}_k - \text{Entry Date}_k)_{\text{days}}$$

#### 18. Daily Mark-to-Market (MTM) Valuation
On each daily bar $t$:
$$\text{Portfolio Value}_t = \text{Cash}_t + \sum_{s \in \text{Open Positions}} \left(\text{Quantity}_s \times \text{Close Price}_{s,t}\right)$$

---

## 8. Web Dashboard & API

The engine includes a web dashboard running on `http://localhost:8000`:
* **Signal Selector**: Instantly pick any signal file (`signalsmain1.py.csv`, `signalsmain2.py.csv`) or upload custom CSVs.
* **Interactive Charts**:
  * Portfolio Equity Curve with High-Watermark peak tracking.
  * Underwater Drawdown percentage area chart.
  * Year-by-Year % Profit histogram.
* **Data Tabs**:
  * **Top 20 Performance Metrics Grid**: Curated institutional layout with one-click **Export CSV** download.
  * **Trade Logs & Execution Records**: Searchable and filterable table with full trade audit trail.
  * **Year-by-Year Performance**: Table breaking down annual return %, realized P&L, trades exited, and ending portfolio equity.
  * **AI / LLM Validation & Multi-Agent Verification**: Side-by-side reconciliation of engine metrics versus LLM audit findings with narrative analysis.

---

## 9. Peer Verification & Reproducibility Checklist

When comparing backtest results with a peer engine, ensure the following parameters are identical:

| Checkpoint | Expected Engine Setting |
| :--- | :--- |
| **Initial Capital** | ₹1,00,00,000 (₹1 Crore INR) |
| **Position Sizing** | 5.0% of *Available Cash* at time of buy |
| **Share Quantities** | Integer shares only (`math.floor`), no fractional shares |
| **Execution Timing** | **Same-day Execution** (Signal generated on Day $T$ $\rightarrow$ Filled on Day $T$) |
| **Slippage** | 0.05% on Buy (Price $\times$ 1.0005) and 0.05% on Sell (Price $\times$ 0.9995) |
| **Brokerage** | 0.03% on both Buy and Sell turnover |
| **STT** | 0.10% on **Sell turnover only** (Buy side = 0.0%) |
| **Stamp Duty** | 0.015% on **Buy turnover only** (Sell side = 0.0%) |
| **GST** | 18% on Brokerage |
| **Terminal Exit** | Open trades on final bar force-closed at final bar Close price (`End of Data`) |
| **Risk-Free Rate** | 6.0% annual (used for Sharpe and Sortino) |
| **Trading Days / Year**| 252 days |

---

## 10. Running the Application

### 1. Launch the Unified Web Dashboard Terminal
```powershell
python src/app.py
```
Server runs at `http://localhost:8000` with live terminal request logging.

### 2. Run the Automated Pytest Suite
```powershell
pytest tests/
```
All 49 unit and integration tests pass deterministically.

---
*Engineered for institutional Indian equity quant research.*
