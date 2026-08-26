# Experiment Engine & Research Suite

## Overview

The Experiment Engine (`src/experiments/`) allows quantitative researchers to design, execute, and analyze reproducible equity strategy experiments without overfitting or look-ahead bias.

---

## 1. Experiment Framework (`src/experiments/runner.py`)

Every experiment run is assigned a unique, immutable experiment ID (e.g. `EXP_6a7eb5a6`) and stores:
- **Hypothesis**: Plain-language investment thesis.
- **Strategy & Parameters**: Concrete parameter configuration.
- **Universe & Date Range**: Tickers and date window.
- **Transaction Assumptions**: Capital, commission rate, slippage, position sizing.
- **Portfolio Constraints**: Liquidity thresholds, max positions cap, rebalancing schedule.
- **Calculated Metrics**: Complete quantitative performance dictionary.
- **Benchmark Metrics**: Nifty 50 relative performance.
- **Validation Results**: Out-of-sample and robustness evaluation.
- **Research Warnings**: Automatically flagged quantitative risks.

---

## 2. Parameter Sweeps (`src/experiments/parameter_sweep.py`)

- Performs systematic multi-parameter grid search across strategy parameters.
- Records CAGR, Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Turnover, Exposure, and Benchmark Excess Return for every combination.
- Exports complete tabular results to DataFrame and CSV.

### Research Rule:
Parameter sweeps are for research and sensitivity analysis. The engine **never** automatically declares the highest Sharpe ratio parameter combination as the "final" live strategy.

---

## 3. Robustness & Sensitivity Analysis (`src/experiments/robustness.py`)

Evaluates strategy P&L sensitivity under small perturbations:
- **Execution Cost Variations**: 0.05%, 0.10%, 0.20% commission rates.
- **Slippage Variations**: 0.02%, 0.05%, 0.10%.
- **Position Size Variations**: 10%, 20%, 50% capital allocation per trade.
- **Parameter Variations**: +/- 10% parameter adjustments.

Computes the **Sharpe Ratio Coefficient of Variation (CV)**:
$$\text{CV}_{\text{Sharpe}} = \frac{\sigma_{\text{Sharpe}}}{\mu_{\text{Sharpe}}}$$
If $\text{CV}_{\text{Sharpe}} < 0.35$, the strategy is flagged as robust.

---

## 4. Market Regime Breakdown (`src/experiments/regime_analysis.py`)

Classifies daily market conditions using Nifty 50 benchmark momentum:
- **Bull Regime**: 20-day return > +2.0%
- **Bear Regime**: 20-day return < -2.0%
- **Sideways Regime**: 20-day return between -2.0% and +2.0%

Evaluates strategy return, volatility, Sharpe ratio, and market exposure separately per regime to identify structural regime dependencies.
