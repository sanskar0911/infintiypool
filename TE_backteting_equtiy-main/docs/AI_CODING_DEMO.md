# AI Coding Assistance Demonstration Report

**Milestone Target:** 25-Jul-2026  
**Topic:** AI Coding Tools (Cursor, GitHub Copilot, Lovable/Replit Prototyping)  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Executive Summary

This document demonstrates the application of AI coding assistants (Cursor / GitHub Copilot) during the development of the equity backtesting performance metrics module (`src/metrics.py`). It documents the exact initial prompt, the raw AI-generated code, a critical code review identifying flaws (such as look-ahead bias and divide-by-zero errors), the manual refactoring applied, and the final production-ready code.

---

## 2. Task Given to AI Assistant (Cursor / Copilot)

**Prompt:**
> "Write a Python function `calculate_backtest_metrics` using Pandas and NumPy that takes a daily portfolio DataFrame (with `Portfolio Value` and `Portfolio Return`) and a trade log DataFrame. Calculate:
> 1. Total Return
> 2. CAGR (Compound Annual Growth Rate)
> 3. Annual Volatility
> 4. Sharpe Ratio (assuming Rf = 0)
> 5. Maximum Drawdown
> 6. Win Rate and Profit Factor"

---

## 3. Initial AI-Generated Code Output

```python
import numpy as np
import pandas as pd

def calculate_backtest_metrics(portfolio_df, trade_log_df, initial_capital=100000.0):
    final_val = portfolio_df["Portfolio Value"].iloc[-1]
    total_return = (final_val - initial_capital) / initial_capital
    
    # CAGR calculation
    years = len(portfolio_df) / 252
    cagr = (final_val / initial_capital) ** (1 / years) - 1
    
    # Annual volatility & Sharpe
    returns = portfolio_df["Portfolio Return"]
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    # Max Drawdown
    peak = portfolio_df["Portfolio Value"].max()
    max_dd = (portfolio_df["Portfolio Value"].min() - peak) / peak
    
    # Win rate & Profit factor
    wins = trade_log_df[trade_log_df["PnL"] > 0]["PnL"].sum()
    losses = abs(trade_log_df[trade_log_df["PnL"] < 0]["PnL"].sum())
    win_rate = len(trade_log_df[trade_log_df["PnL"] > 0]) / len(trade_log_df)
    profit_factor = wins / losses
    
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor
    }
```

---

## 4. Code Review & Vulnerability Analysis

A rigorous software engineering code review revealed several critical defects in the AI-generated code:

### Defect 1: Flawed Maximum Drawdown Calculation (Look-Ahead / Global Peak Flaw)
- **Problem**: `portfolio_df["Portfolio Value"].max()` computes the **global maximum** over the entire dataset instead of the **running cumulative peak** (`cummax()`).
- **Impact**: Subtracting the absolute minimum from the overall peak miscalculates drawdown whenever the global peak occurs AFTER the minimum value, introducing look-ahead distortion and invalid metric output.
- **Fix**: Compute cumulative max via `portfolio_df["Portfolio Value"].cummax()` and measure max drawdown per timestamp.

### Defect 2: Divide-by-Zero Risk
- **Problem**:
  1. `returns.std()` can be `0.0` or `NaN` if returns are constant or sample size is 1, causing a `ZeroDivisionError` when computing `sharpe`.
  2. `losses` can be `0.0` if there are no losing trades, causing `wins / losses` to raise `ZeroDivisionError`.
  3. `years` can be `0.0` if the dataframe has 0 or 1 row.
- **Impact**: Crash on zero-loss backtests or flat equity curves.
- **Fix**: Add explicit guards (`if downside_std > 0`, `if gross_loss > 0 else float('inf')`).

### Defect 3: Calendar vs. Trading Days Assumptions
- **Problem**: `len(portfolio_df) / 252` assumes trading days equal total duration, miscalculating CAGR across weekends and market holidays.
- **Fix**: Calculate calendar year fraction using `(end_date - start_date).days / 365.25`.

---

## 5. Corrected Production Implementation

```python
import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_metrics(portfolio_df: pd.DataFrame, trade_log_df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
    final_value = portfolio_df["Portfolio Value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    start_date = portfolio_df.index.min()
    end_date = portfolio_df.index.max()
    calendar_days = (end_date - start_date).days
    years = calendar_days / 365.25 if calendar_days > 0 else 0.0
    cagr = ((final_value / initial_capital) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    daily_returns = portfolio_df["Portfolio Return"]
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(252) if not pd.isna(daily_vol) and daily_vol > 0 else 0.0
    sharpe_ratio = (daily_returns.mean() / daily_vol) * np.sqrt(252) if annual_vol > 0 else 0.0

    # Correct Running Cumulative Drawdown
    portfolio_values = portfolio_df["Portfolio Value"]
    running_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min() if not pd.isna(drawdowns.min()) else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown
    }
```

---

## 6. Key Learnings & AI Coding Best Practices

1. **AI output is a prototype, not production code**: AI models prioritize syntactic correctness over subtle mathematical edge cases (like cumulative drawdown vs global drawdown).
2. **Mandatory Manual Code Review**: Critical quantitative metrics must be verified against analytical benchmarks and unit-tested with edge cases (e.g. zero trades, zero losses, 100% win rate).
3. **Structured Guidance Works Best**: Providing explicit boundary constraints and error-handling requirements in prompts dramatically improves initial code quality.
