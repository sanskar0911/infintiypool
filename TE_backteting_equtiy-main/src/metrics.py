"""
metrics.py

Comprehensive Quantitative Performance & Risk Metrics Library for QUANTLAB v1.1.
Implements mathematically authoritative calculations for portfolio and trade-level KPIs:
- Net Profit & Total Return (%)
- CAGR (%) with elapsed calendar days annualization
- Win Rate, Gross Profit, Gross Loss, Profit Factor
- Average Trade, Average Win, Average Loss, Expectancy (EV)
- Mark-to-Market Daily Returns, Annualized Volatility (%)
- Sharpe Ratio (Rf = 0 baseline)
- Sortino Ratio (downside deviation over all daily observations)
- Max Drawdown (INR & %) and Calmar Ratio
- Max Win/Loss streaks and skewness
- Robust handling for zero-trade, zero-volatility, and zero-loss edge cases.
"""

from typing import Dict, List, Any, Optional
import math
import numpy as np
import pandas as pd


def calculate_metrics(
    portfolio_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    initial_capital: float = 10000000.0,
    risk_free_rate: float = 0.0,
    annualization_days: int = 252,
    price_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive quantitative KPIs for the backtested portfolio.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Daily portfolio tracking data containing "Portfolio Value" and optional "Portfolio Return".
    trade_log_df : pd.DataFrame
        Log of executed trades and orders.
    initial_capital : float
        Starting cash balance in INR (default: ₹10,000,000).
    risk_free_rate : float
        Annualized risk-free rate (default: 0.0).
    annualization_days : int
        Number of trading days per year (default: 252).
    price_df : pd.DataFrame, optional
        Stock price dataframe.

    Returns
    -------
    Dict[str, Any]
        Calculated metric dictionary adhering to QUANTLAB v1.1 specifications.
    """
    # --------------------------------------------------------------------------
    # 1. Equity & Returns Calculations
    # --------------------------------------------------------------------------
    if portfolio_df is not None and not portfolio_df.empty and "Portfolio Value" in portfolio_df.columns:
        equity_series = portfolio_df["Portfolio Value"].dropna()
        final_equity = float(equity_series.iloc[-1])
        start_date = portfolio_df.index.min()
        end_date = portfolio_df.index.max()
        calendar_days = max(1, (end_date - start_date).days)
        years = calendar_days / 365.25
    else:
        equity_series = pd.Series([initial_capital])
        final_equity = initial_capital
        calendar_days = 0
        years = 0.0

    net_profit = final_equity - initial_capital
    total_return_pct = 100.0 * (final_equity / initial_capital - 1.0) if initial_capital > 0 else 0.0

    # CAGR %
    if years > 0 and final_equity > 0 and initial_capital > 0:
        cagr_pct = 100.0 * ((final_equity / initial_capital) ** (365.25 / calendar_days) - 1.0)
    else:
        cagr_pct = 0.0

    # Daily MTM Returns & Volatility
    if len(equity_series) > 1:
        daily_returns = equity_series.pct_change().dropna()
    else:
        daily_returns = pd.Series(dtype=float)

    n_daily_obs = len(daily_returns)
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_days) - 1.0 if risk_free_rate > 0 else 0.0

    if n_daily_obs > 1:
        daily_vol = float(daily_returns.std(ddof=1))
        annualized_vol_pct = 100.0 * daily_vol * math.sqrt(annualization_days) if daily_vol > 0 else 0.0
    else:
        daily_vol = 0.0
        annualized_vol_pct = 0.0

    # Sharpe Ratio
    if daily_vol > 1e-9:
        excess_returns = daily_returns - daily_rf
        mean_excess = float(excess_returns.mean())
        sharpe_ratio = float(math.sqrt(annualization_days) * mean_excess / daily_vol)
    else:
        sharpe_ratio = None if n_daily_obs <= 1 else 0.0

    # Downside Deviation & Sortino Ratio
    # Definition: sqrt(mean(min(r_d - r_f, 0)^2)) using all observations in denominator
    if n_daily_obs > 0:
        excess_returns = daily_returns - daily_rf
        downside_diffs = np.minimum(excess_returns, 0.0)
        downside_variance = np.mean(downside_diffs ** 2)
        downside_dev = math.sqrt(float(downside_variance))
        if downside_dev > 1e-9:
            sortino_ratio = float(math.sqrt(annualization_days) * float(excess_returns.mean()) / downside_dev)
        else:
            sortino_ratio = float("inf") if excess_returns.mean() > 0 else None
    else:
        downside_dev = 0.0
        sortino_ratio = None

    # Maximum Drawdown
    if len(equity_series) > 0:
        running_peak = equity_series.cummax()
        drawdown_amounts = running_peak - equity_series
        drawdown_pcts = 100.0 * (drawdown_amounts / running_peak)
        max_drawdown_amount = float(drawdown_amounts.max()) if not drawdown_amounts.empty else 0.0
        max_drawdown_pct = float(drawdown_pcts.max()) if not drawdown_pcts.empty else 0.0
    else:
        max_drawdown_amount = 0.0
        max_drawdown_pct = 0.0

    # Calmar Ratio: CAGR % / Max Drawdown %
    if max_drawdown_pct > 1e-6:
        calmar_ratio = float(cagr_pct / max_drawdown_pct)
    else:
        calmar_ratio = float("inf") if cagr_pct > 0 else None

    # --------------------------------------------------------------------------
    # 2. Closed-Trade Level Performance
    # --------------------------------------------------------------------------
    closed_pnls: List[float] = []
    closed_pnl_pcts: List[float] = []
    holding_periods: List[int] = []

    if trade_log_df is not None and not trade_log_df.empty:
        # Check if Realized PnL is already recorded row-by-row
        if "Realized PnL" in trade_log_df.columns:
            sell_rows = trade_log_df[trade_log_df["Realized PnL"] != 0.0]
            if not sell_rows.empty:
                closed_pnls = sell_rows["Realized PnL"].astype(float).tolist()

        # If not populated via Realized PnL column, match sequential BUYs and SELLs
        if not closed_pnls:
            type_col = "BUY/SELL" if "BUY/SELL" in trade_log_df.columns else "Type"
            if type_col in trade_log_df.columns:
                type_str = trade_log_df[type_col].astype(str).str.upper()
                buys = trade_log_df[type_str.str.contains("BUY")]
                sells = trade_log_df[type_str.str.contains("SELL")]
                n_pairs = min(len(buys), len(sells))

                for idx in range(n_pairs):
                    b_row = buys.iloc[idx]
                    s_row = sells.iloc[idx]

                    b_qty = float(b_row.get("Shares", b_row.get("Quantity", 1)))
                    s_qty = float(s_row.get("Shares", s_row.get("Quantity", 1)))
                    qty = min(b_qty, s_qty)

                    b_px = float(b_row.get("Execution Price", b_row.get("Executed Price", b_row.get("Price", 0.0))))
                    s_px = float(s_row.get("Execution Price", s_row.get("Executed Price", s_row.get("Price", 0.0))))

                    b_cost = float(b_row.get("Transaction Cost", b_row.get("Commission", 0.0)))
                    s_cost = float(s_row.get("Transaction Cost", s_row.get("Commission", 0.0)))

                    trade_pnl = (qty * s_px) - (qty * b_px) - b_cost - s_cost
                    closed_pnls.append(trade_pnl)
                    total_entry_val = qty * b_px + b_cost
                    if total_entry_val > 0:
                        closed_pnl_pcts.append(trade_pnl / total_entry_val)

    total_closed_trades = len(closed_pnls)
    winning_pnls = [p for p in closed_pnls if p > 0]
    losing_pnls = [p for p in closed_pnls if p < 0]
    even_pnls = [p for p in closed_pnls if p == 0]

    winning_count = len(winning_pnls)
    losing_count = len(losing_pnls)
    even_count = len(even_pnls)

    win_rate_pct = 100.0 * (winning_count / total_closed_trades) if total_closed_trades > 0 else 0.0
    loss_rate_pct = 100.0 * (losing_count / total_closed_trades) if total_closed_trades > 0 else 0.0

    gross_profit = float(sum(winning_pnls))
    gross_loss = float(abs(sum(losing_pnls)))

    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None if total_closed_trades == 0 else 0.0

    avg_trade_pnl = float(sum(closed_pnls) / total_closed_trades) if total_closed_trades > 0 else 0.0
    avg_win_pnl = float(gross_profit / winning_count) if winning_count > 0 else 0.0
    avg_loss_pnl = float(gross_loss / losing_count) if losing_count > 0 else 0.0

    # Expectancy (EV) = Win_prob * Average_win - Loss_prob * Average_loss
    win_prob = winning_count / total_closed_trades if total_closed_trades > 0 else 0.0
    loss_prob = losing_count / total_closed_trades if total_closed_trades > 0 else 0.0
    expectancy_ev = (win_prob * avg_win_pnl) - (loss_prob * avg_loss_pnl) if total_closed_trades > 0 else 0.0

    # Streaks & Skewness
    max_win_streak = 0
    max_loss_streak = 0
    curr_win = 0
    curr_loss = 0
    for p in closed_pnls:
        if p > 0:
            curr_win += 1
            curr_loss = 0
            if curr_win > max_win_streak:
                max_win_streak = curr_win
        elif p < 0:
            curr_loss += 1
            curr_win = 0
            if curr_loss > max_loss_streak:
                max_loss_streak = curr_loss
        else:
            curr_win = 0
            curr_loss = 0

    if len(closed_pnls) >= 3 and np.std(closed_pnls) > 0:
        pnl_series = pd.Series(closed_pnls)
        skewness_val = float(pnl_series.skew())
    else:
        skewness_val = 0.0

    # Format output dictionary with both descriptive and numerical representations
    return {
        # Portfolio Equity & Returns
        "Initial Capital": initial_capital,
        "Final Equity": final_equity,
        "Net Profit": net_profit,
        "Total Return": total_return_pct / 100.0,
        "Total Return %": total_return_pct,
        "CAGR": cagr_pct / 100.0,
        "CAGR %": cagr_pct,
        "Calendar Days": calendar_days,
        "Years": years,

        # Risk-Adjusted Ratios
        "Sharpe Ratio": sharpe_ratio if sharpe_ratio is not None else 0.0,
        "Sortino Ratio": sortino_ratio if sortino_ratio is not None else 0.0,
        "Annualized Volatility %": annualized_vol_pct,
        "Downside Deviation": downside_dev,
        "Maximum Drawdown": -max_drawdown_pct / 100.0,
        "Maximum Drawdown %": -max_drawdown_pct,
        "Maximum Drawdown INR": -max_drawdown_amount,
        "Calmar Ratio": calmar_ratio if calmar_ratio is not None else 0.0,

        # Trade-Level Statistics
        "Total Closed Trades": total_closed_trades,
        "Winning Trades": winning_count,
        "Losing Trades": losing_count,
        "Even Trades": even_count,
        "Win Rate": win_rate_pct / 100.0,
        "Win Rate %": win_rate_pct,
        "Loss Rate %": loss_rate_pct,
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Profit Factor": profit_factor if profit_factor is not None else 0.0,
        "Average Trade PnL": avg_trade_pnl,
        "Average Win": avg_win_pnl,
        "Average Loss": avg_loss_pnl,
        "Expectancy (EV)": expectancy_ev,
        "Max Win Streak": max_win_streak,
        "Max Loss Streak": max_loss_streak,
        "Skewness": skewness_val,
        "Exposure": float(portfolio_df["Exposure"].mean()) if (portfolio_df is not None and "Exposure" in portfolio_df.columns) else 0.0,
        "Turnover": 0.0
    }
