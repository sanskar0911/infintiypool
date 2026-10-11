# QUANTLAB Reports Directory

This directory contains the central project documentation, system architecture specifications, and execution reports for the QUANTLAB Equity Strategy Backtesting Framework.

---

## Directory Contents

| File / Folder | Description |
| :--- | :--- |
| [`MASTER_PROJECT_REPORT.md`](./MASTER_PROJECT_REPORT.md) | **Master Project System Architecture & Validation Report**. Contains complete system overview, Mermaid architecture diagrams, order execution lifecycle flowchart, LangGraph agent state machine, 8-stage MADRID RAG chain diagram, quantitative KPI formulas, and automated test suite audit. |

---

## System Outputs Overview

In addition to documentation, the backtesting engine and reporting tools generate output files in the following standard locations:

- **`results/`**: Standard execution output folder containing:
  - `summary_report.txt` (Strategy Fact Sheet text output)
  - `signals_chart.png` (Technical strategy signal chart)
  - `equity_curve.png` (Portfolio value growth curve)
  - `drawdown.png` (Underwater drawdown curve)
  - `rolling_sharpe.png` (60-day rolling Sharpe ratio chart)
  - `monthly_returns.png` (Monthly return bar breakdown)
  - `trade_dist.png` (Trade return distribution histogram)
  - `allocation.png` (Cash vs Equity asset allocation stacked area)
  - `trade_log.csv` & `portfolio.csv` (Time-series data tables)

- **`output/`**: Institutional Excel report folder containing:
  - `backtest_report.xlsx` (10-sheet Master Workbook with Summary, Trade Log, Monthly Returns, Yearly Returns, Equity Curve, Drawdown, Metrics, Input Validation, Excel vs Python Reconciliation, and LLM Validation).

---

*QUANTLAB Quantitative Systems Engineering Team.*
