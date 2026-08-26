# System Architecture & Technical Specifications

## Shankh / Decuple Internship 2026-27 (Project TE1)
### Project: Backtesting Equity Strategies for Indian Stocks

---

## 1. System Overview

The framework is an institutional-grade, research-oriented quantitative equity backtesting and research environment specifically calibrated for Indian equities (NSE tickers).

It enforces strict modular separation across data ingestion, signal generation, portfolio execution, risk management, benchmark relative attribution, out-of-sample validation, and autonomous research orchestration via LangGraph.

---

## 2. Core Architectural Layers

```
                               ┌───────────────────────────┐
                               │     Streamlit UI App      │
                               └─────────────┬─────────────┘
                                             │
                               ┌─────────────▼─────────────┐
                               │  LangGraph Agent Workflow │
                               │    (Loop Engineering)     │
                               └─────────────┬─────────────┘
                                             │
      ┌────────────────────────┬─────────────┴─────────────┬────────────────────────┐
      │                        │                           │                        │
┌─────▼──────────┐   ┌─────────▼─────────┐       ┌─────────▼─────────┐    ┌─────────▼─────────┐
│ Data Loader &  │   │ Strategy Engine & │       │ Execution Layer   │    │  Metrics &        │
│ Data Cleaner   │   │ Factory Pattern   │       │ & Portfolio Rules │    │  Benchmark Engine │
└─────┬──────────┘   └─────────┬─────────┘       └─────────┬─────────┘    └─────────┬─────────┘
      │                        │                           │                        │
      └────────────────────────┴─────────────┬─────────────┴────────────────────────┘
                                             │
                               ┌─────────────▼─────────────┐
                               │ Experiment Suite & Audit  │
                               │ (Walk-Forward, OOS, Rob)  │
                               └───────────────────────────┘
```

### 1. Data Ingestion & Data Hygiene Layer
- **Modules**: `src/data_loader.py`, `src/clean_data.py`
- Downloads raw daily OHLCV CSV data for NSE equities and Nifty 50 benchmark (`^NSEI`).
- Disables automatic dividend/split adjustments during initial fetch (`auto_adjust=False`) to maintain raw Close and Adjusted Close transparency.
- Enforces chronological index sorting, removes duplicate rows, forward-fills price gaps without volume fabrication, and derives standard returns.

### 2. Strategy Engine Layer (Factory Pattern)
- **Modules**: `src/strategy.py`, `src/strategies/base.py`, `src/strategies/factory.py`
- Concrete strategy classes (`SMACrossoverStrategy`, `EMACrossoverStrategy`, `RSIStrategy`) inherit from abstract `BaseStrategy`.
- Uses `StrategyFactory` for dynamic instantiation and parameter configuration.

### 3. Execution & Portfolio Constraint Layer
- **Modules**: `src/backtester.py`, `src/portfolio.py`
- Handles slippage modeling (0.05% markup on buys, discount on sells), commission rates (0.1%), fractional position allocation, stop-loss triggers, and take-profit triggers.
- **Liquidity Filter**: Enforces minimum daily volume, minimum traded value (e.g. INR 5 Lakhs), and minimum stock price.
- **Maximum Positions Cap**: Applies deterministic ranking (e.g. momentum rank) when active signals exceed max permitted positions (`max_positions=10`).
- **Rebalancing Schedule**: Enforces Daily, Weekly, or Monthly rebalancing dates with zero future leakage.

### 4. Quantitative Metrics & Benchmark Engine
- **Modules**: `src/metrics.py`, `src/benchmark.py`
- Computes CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown, Win Rate, Profit Factor, Turnover, Market Exposure, and Daily Hit Ratio.
- Aligns strategy portfolio returns with Nifty 50 benchmark returns to compute Excess Return, Beta, Correlation, Tracking Error, and Information Ratio.

### 5. Experiment Suite & Validation Engine
- **Modules**: `src/experiments/` (`runner.py`, `parameter_sweep.py`, `walk_forward.py`, `out_of_sample.py`, `robustness.py`, `regime_analysis.py`, `research_warnings.py`)
- Provides systematic grid search, rolling walk-forward validation, train/test out-of-sample splits, look-ahead bias audit, sensitivity perturbations, market regime performance breakdown, and automated quantitative research warnings.

### 6. LangGraph Agent Workflow (Loop Engineering)
- **Module**: `src/agent/workflow.py`
- Implements a research-oriented controlled state graph (`HYPOTHESIS -> BASELINE -> BACKTEST -> EVALUATE -> ITERATE / VALIDATE -> REPORT`).
- Evaluates strategy against benchmark and performance targets with strict stopping conditions (max iterations, overfitting warnings, stability checks).
