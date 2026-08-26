# Validation Methodology & Bias Prevention

## Overview

A backtest is worthless if it suffers from look-ahead bias, survivorship bias, data leakage, or overfitting. This document details the quantitative validation protocols implemented in the framework.

---

## 1. Look-Ahead Bias Prevention (`src/experiments/out_of_sample.py`)

Look-ahead bias occurs when future price data is inadvertently used to generate today's signal or execute today's trade.

### Automated Look-Ahead Bias Audit:
- **Chronological Sorting Verification**: Confirms that price data indices are strictly monotonically increasing.
- **Index Alignment Verification**: Confirms signal date index matches price date index exactly.
- **Signal-to-Position Entry Verification**: Confirms that position entries occur on or after signal generation dates without future shift.

If any check fails, the experiment is marked as **`FAILED`** and halted immediately.

---

## 2. Out-of-Sample (OOS) Validation

- Divides historical price data into an **In-Sample (IS)** period (e.g. 70% of dates) and an **Out-of-Sample (OOS)** period (30% of dates).
- Strategy parameters calibrated or selected in In-Sample are tested strictly out-of-sample without re-calibration.
- Calculates **Sharpe Degradation Percentage**:
$$\text{Degradation}_{\text{Sharpe}} = \frac{\text{Sharpe}_{\text{OOS}} - \text{Sharpe}_{\text{IS}}}{\text{Sharpe}_{\text{IS}}} \times 100\%$$
- If OOS Sharpe drops by more than 30%, an **Overfitting Warning** is logged.

---

## 3. Walk-Forward Testing (`src/experiments/walk_forward.py`)

- Implements rolling train/test window analysis (e.g. 3-year train, 1-year test, 1-year step).
- Evaluates **Walk-Forward Efficiency (WFE)**:
$$\text{WFE} = \frac{\text{Mean Sharpe}_{\text{OOS}}}{\text{Mean Sharpe}_{\text{IS}}}$$
- A WFE $\ge 0.50$ indicates strong parameter stability across changing market regimes.

---

## 4. Automated Research Warnings (`src/experiments/research_warnings.py`)

The framework automatically scans and logs warnings for:
1. **Insufficient Sample Size** (< 30 completed trades).
2. **High Turnover Risk** (> 300% annual turnover).
3. **Excessive Drawdown** (> 25% peak-to-trough decline).
4. **Suboptimal Sharpe** (< 0.50).
5. **Narrow Profit Factor** (< 1.20).
6. **Benchmark Underperformance** (CAGR < Nifty 50 CAGR).
7. **High Parameter Sensitivity** (CV of Sharpe > 0.35).
8. **Overfitting Evidence** (OOS Sharpe degradation > 30%).
