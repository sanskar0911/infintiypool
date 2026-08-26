# Institutional Equity Quantitative Backtesting & Research Framework

## Shankh / Decuple Internship 2026-27 (Project TE1)
**Project Title:** Backtesting Equity Strategies for Indian Equities

A production-grade, modular Python Equity Backtesting and AI-driven Quantitative Analysis framework for NSE stock equities and Nifty 50 benchmark comparison.

---

## 📁 Complete Repository Structure

```text
TE_backteting_equtiy-main/
│
├── README.md                      # Primary project documentation & architecture guide
├── AI_CODING_REPORT.md            # Engineering report on prompt design & code reviews
├── requirements.txt               # Python package dependencies
├── index.html                     # Web UI static dashboard template
│
├── data/                          # Market data repository
│   ├── raw/                       # Raw daily OHLCV data CSVs (NIFTY50, INFY, RELIANCE, TCS, etc.)
│   └── processed/                 # Cleaned and validated CSV data for backtesting
│       ├── NIFTY50.csv            # Nifty 50 Benchmark dataset
│       ├── INFY.csv
│       ├── RELIANCE.csv
│       ├── TCS.csv
│       ├── HDFCBANK.csv
│       └── ICICIBANK.csv
│
├── docs/                          # Comprehensive Technical Documentation
│   ├── ARCHITECTURE.md            # System architecture & layer specifications
│   ├── EXPERIMENTS.md             # Parameter sweeps & experiment runner guide
│   ├── VALIDATION.md              # Walk-forward, OOS split & bias prevention methodology
│   └── LOOP_ENGINEERING.md        # LangGraph controlled state machine & decision rules
│
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── EDA.ipynb                  # Exploratory Data Analysis & returns correlation notebook
│   └── Backtesting.ipynb          # Step-by-step strategy backtest tutorial notebook
│
├── results/                       # Backtest output artifacts & generated charts
│   ├── signals_chart.png          # Stock price & signal markers plot
│   ├── equity_curve.png           # Portfolio growth vs initial capital plot
│   ├── drawdown.png               # Underwater drawdown chart
│   ├── rolling_sharpe.png         # Rolling 60-day annualized Sharpe Ratio chart
│   ├── monthly_returns.png        # Monthly returns breakdown bar chart
│   ├── trade_dist.png             # Histogram of completed trade returns
│   ├── allocation.png             # Stacked area asset allocation chart (Cash vs Equity)
│   └── summary.txt                # Automated Strategy Fact Sheet & performance report
│
├── tests/                         # Automated Unit Test Suite
│   └── test_backtest_rules.py     # Pytest unit tests for look-ahead bias, costs, limits, splits
│
└── src/                           # Core Source Code Engine
    ├── __init__.py
    ├── main.py                    # Main orchestrator CLI & pipeline runner
    ├── app.py                     # Institutional Streamlit Web Dashboard
    ├── backtester.py              # Execution engine (costs, slippage, position caps, rebalancing)
    ├── portfolio.py               # Liquidity filtering, position limits & rebalancing schedule
    ├── benchmark.py               # Nifty 50 benchmark loader, return & risk comparison engine
    ├── strategy.py                # Facade module delegating to strategy library
    ├── metrics.py                 # Quantitative KPIs & risk metrics calculator
    ├── visualization.py           # Publication-grade chart generation engine
    ├── report.py                  # Final Strategy Fact Sheet generator
    ├── data_loader.py             # YFinance historical market data loader
    ├── clean_data.py              # Data cleaning, missing value handler & validator
    ├── utils.py                   # Shared calculation helper utilities
    │
    ├── strategies/                # Modular Strategy Library (Strategy Factory Pattern)
    │   ├── base.py                # Abstract Base Strategy interface (BaseStrategy)
    │   ├── sma_crossover.py       # Simple Moving Average Crossover Strategy (SMA)
    │   ├── ema_crossover.py       # Exponential Moving Average Crossover Strategy (EMA)
    │   ├── rsi_strategy.py        # Relative Strength Index Strategy (RSI)
    │   └── factory.py             # Strategy Factory class (StrategyFactory)
    │
    ├── experiments/               # Quantitative Experiment Engine & Audit Suite
    │   ├── __init__.py
    │   ├── runner.py              # Reproducible experiment runner & ID tracking
    │   ├── parameter_sweep.py     # Grid search parameter sweep engine
    │   ├── walk_forward.py        # Rolling train/test walk-forward validation
    │   ├── out_of_sample.py       # Out-of-sample split & look-ahead bias audit
    │   ├── robustness.py          # Sensitivity analysis under cost/sizing perturbations
    │   ├── regime_analysis.py     # Market regime breakdown (Bull/Bear/Sideways)
    │   └── research_warnings.py   # Automated quantitative research warnings generator
    │
    ├── llm/                       # LLM Strategy Analysis Engine
    │   ├── prompts.py             # Structured financial prompts & JSON output schemas
    │   └── analyzer.py            # LLM analyzer module (returns JSON rating, risk, confidence)
    │
    └── agent/                     # LangGraph Agent Workflow
        └── workflow.py            # Controlled Loop Engineering research state machine
```

---

## 📄 Core Module Responsibilities

### Core Engine (`src/`)
- **`src/main.py`**: CLI entry point. Executes research pipeline with configurable strategy, capital, commission, position size, SL/TP, liquidity rules, and rebalancing frequency flags.
- **`src/app.py`**: Institutional Streamlit Web Dashboard (`streamlit run src/app.py`). Features sidebar controls, KPI metric cards, analytical charts, trade log tables, experiment suite, LangGraph loop visualizer, and Fact Sheet downloads.
- **`src/backtester.py`**: Simulation engine supporting slippage (0.05%), transaction commission (0.1%), liquidity checks, max position caps, stop loss, take profit, and rebalance schedules.
- **`src/portfolio.py`**: Enforces `LiquidityFilter` (min daily volume/traded value/price), `PositionAllocator` (deterministic ranking when signals > limit), and `RebalanceSchedule` (Daily, Weekly, Monthly).
- **`src/benchmark.py`**: Loads Nifty 50 benchmark data, computes benchmark return/CAGR/Sharpe/MDD, and calculates Excess Return, Beta, Correlation, Tracking Error, and Information Ratio.
- **`src/metrics.py`**: Calculates CAGR, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor, Turnover, Exposure, and Daily Hit Ratio.
- **`src/report.py`**: Generates institutional **Final Strategy Fact Sheet** documenting specifications, risk metrics, trading statistics, benchmark comparison, and research warnings.

### Research & Experiments (`src/experiments/`)
- **`runner.py`**: Reproducible experiment class with unique experiment ID tracking.
- **`parameter_sweep.py`**: Systematic grid search saving full comparison tables without auto-declaring top Sharpe.
- **`walk_forward.py`**: Rolling train/test window analysis evaluating parameter and performance stability.
- **`out_of_sample.py`**: In-Sample vs Out-of-Sample split and automated look-ahead bias audit.
- **`robustness.py`**: Perturbation sensitivity testing under execution cost, slippage, and position sizing variations.
- **`regime_analysis.py`**: Market regime classification (Bull, Bear, Sideways) and regime-wise metrics.
- **`research_warnings.py`**: Automated detection of overfitting, data leakage, high turnover, drawdown risk, and benchmark underperformance.

### LangGraph Agent Workflow (`src/agent/workflow.py`)
- Implements **Loop Engineering** research state machine (`HYPOTHESIS -> BASELINE -> BACKTEST -> METRICS & BENCHMARK -> RESEARCH EVALUATOR -> ITERATE / VALIDATE -> REPORT`).
- Evaluates strategy performance against benchmark with strict stopping conditions (max iterations, overfitting warnings, stability checks).

---

## 🚀 Execution & Usage Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Test Suite
```bash
pytest tests/test_backtest_rules.py
```

### 3. Run Main CLI Pipeline
```bash
python src/main.py RELIANCE --strategy SMA --position-size 0.2 --stop-loss 0.05 --take-profit 0.10
```

### 4. Launch Institutional Streamlit Dashboard
```bash
streamlit run src/app.py
```

### 5. Execute LangGraph Controlled Agent Research Workflow
```bash
python src/agent/workflow.py
```

---

## 📊 Final Requirements Matrix

| Requirement | Implementation | File Path | Status |
| :--- | :--- | :--- | :--- |
| **Benchmark Comparison** | Nifty 50 load, alignment, excess return, Beta, Correlation, Information Ratio | [src/benchmark.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/benchmark.py) | ✅ Complete & Tested |
| **Liquidity Filtering** | Min daily volume, min traded value, min price checks before entry | [src/portfolio.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/portfolio.py) | ✅ Complete & Tested |
| **Maximum Positions** | Configurable max simultaneous positions with deterministic momentum ranking | [src/portfolio.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/portfolio.py) | ✅ Complete & Tested |
| **Rebalancing Frequency** | Daily, Weekly, Monthly rebalancing schedules with zero look-ahead leakage | [src/portfolio.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/portfolio.py) | ✅ Complete & Tested |
| **Metrics Engine Extension**| CAGR, Volatility, Sharpe, Sortino, Calmar, MDD, Win Rate, Turnover, Exposure | [src/metrics.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/metrics.py) | ✅ Complete & Tested |
| **Experiment Engine** | Reproducible experiment tracking with unique experiment IDs | [src/experiments/runner.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/runner.py) | ✅ Complete & Tested |
| **Parameter Sweeps** | Systematic grid search exporting complete metrics table to CSV/DataFrame | [src/experiments/parameter_sweep.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/parameter_sweep.py) | ✅ Complete & Tested |
| **Walk-Forward Testing** | Rolling train/test windows, parameter stability, walk-forward efficiency | [src/experiments/walk_forward.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/walk_forward.py) | ✅ Complete & Tested |
| **Out-of-Sample Validation**| Train/Test date splits and automated look-ahead data leakage audit | [src/experiments/out_of_sample.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/out_of_sample.py) | ✅ Complete & Tested |
| **Robustness Analysis** | Sensitivity perturbations across transaction costs, slippage, and position sizes | [src/experiments/robustness.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/robustness.py) | ✅ Complete & Tested |
| **Regime Analysis** | Market regime classification (Bull, Bear, Sideways) and regime-wise metrics | [src/experiments/regime_analysis.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/regime_analysis.py) | ✅ Complete & Tested |
| **Research Warnings** | Automated warnings for overfitting, data leakage, high turnover, drawdown, etc. | [src/experiments/research_warnings.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/experiments/research_warnings.py) | ✅ Complete & Tested |
| **Loop Engineering** | LangGraph research state machine with controlled decision rules and stopping logic | [src/agent/workflow.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/agent/workflow.py) | ✅ Complete & Tested |
| **Final Strategy Fact Sheet**| Institutional summary report detailing strategy specs, risk, benchmark & warnings | [src/report.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/report.py) | ✅ Complete & Tested |
| **Streamlit Dashboard** | Professional UI with Benchmark, Experiments & LangGraph Loop visualizer | [src/app.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/src/app.py) | ✅ Complete & Tested |
| **Automated Tests** | 9 unit tests covering bias, costs, limits, splits, and stopping conditions | [tests/test_backtest_rules.py](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/tests/test_backtest_rules.py) | ✅ Complete & Tested |
| **Documentation** | Technical documentation for architecture, experiments, validation, and loop engineering | [docs/](file:///c:/Users/sanskar%20jagdish/OneDrive/Desktop/infinitypool/TE_backteting_equtiy-main/docs/) | ✅ Complete & Tested |

---

*Engineered with Python, Streamlit, Matplotlib, Pytest, LangChain, and LangGraph.*
