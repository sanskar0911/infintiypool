"""
metrics.py

Comprehensive performance and risk metrics computation module.
Computes Total Return, CAGR, Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio,
Max Drawdown, Win Rate, Profit Factor, Average Holding Period, Average Trade Return,
Turnover, Market Exposure, and Daily Hit Ratio.
"""

import os
import sys
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def calculate_metrics(
    portfolio_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    price_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive quantitative KPIs for the backtested portfolio.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Daily portfolio tracking data containing "Portfolio Value" and "Portfolio Return".
    trade_log_df : pd.DataFrame
        Log of executed trades.
    initial_capital : float
        Starting cash balance in INR.
    price_df : pd.DataFrame, optional
        Stock price dataframe.

    Returns
    -------
    Dict[str, Any]
        Calculated metric dictionary.
    """
    final_value = portfolio_df["Portfolio Value"].iloc[-1]
    
    # 1. Total Return
    total_return = (final_value - initial_capital) / initial_capital

    # 2. CAGR (Compound Annual Growth Rate)
    start_date = portfolio_df.index.min()
    end_date = portfolio_df.index.max()
    calendar_days = (end_date - start_date).days
    years = calendar_days / 365.25 if calendar_days > 0 else 0.0
    cagr = ((final_value / initial_capital) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    # 3. Annual Volatility
    daily_returns = portfolio_df["Portfolio Return"]
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(252) if not pd.isna(daily_vol) and daily_vol > 0 else 0.0

    # 4. Sharpe Ratio (Rf = 0)
    mean_daily_return = daily_returns.mean()
    sharpe_ratio = (mean_daily_return / daily_vol) * np.sqrt(252) if annual_vol > 0 else 0.0

    # 5. Sortino Ratio (Downside volatility, Rf = 0)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        downside_std = negative_returns.std()
        sortino_ratio = (mean_daily_return / downside_std) * np.sqrt(252)
    else:
        sortino_ratio = 0.0

    # 6. Maximum Drawdown
    portfolio_values = portfolio_df["Portfolio Value"]
    running_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min() if not pd.isna(drawdowns.min()) else 0.0

    # 7. Calmar Ratio
    calmar_ratio = (cagr / abs(max_drawdown)) if (max_drawdown < 0 and cagr > 0) else 0.0

    # 8. Trade-level Analysis
    trade_returns: List[float] = []
    holding_periods: List[int] = []
    gross_profit = 0.0
    gross_loss = 0.0

    if not trade_log_df.empty:
        # Filter buys and sells
        type_col = "BUY/SELL" if "BUY/SELL" in trade_log_df.columns else "Type"
        buys = trade_log_df[trade_log_df[type_col] == "BUY"]
        sells = trade_log_df[trade_log_df[type_col] == "SELL"]
        n_trades = min(len(buys), len(sells))
        
        for idx in range(n_trades):
            buy_row = buys.iloc[idx]
            sell_row = sells.iloc[idx]
            
            buy_date = buy_row.name if isinstance(buy_row.name, pd.Timestamp) else pd.to_datetime(buy_row.name)
            sell_date = sell_row.name if isinstance(sell_row.name, pd.Timestamp) else pd.to_datetime(sell_row.name)
            
            days_held = (sell_date - buy_date).days
            holding_periods.append(max(days_held, 1))

            buy_val = buy_row["Shares"] * buy_row["Execution Price"]
            buy_fee = buy_row.get("Commission", buy_row.get("Transaction Cost", 0.0))
            total_buy_cost = buy_val + buy_fee

            sell_val = sell_row["Shares"] * sell_row["Execution Price"]
            sell_fee = sell_row.get("Commission", sell_row.get("Transaction Cost", 0.0))
            net_sell_proceeds = sell_val - sell_fee

            pnl = net_sell_proceeds - total_buy_cost
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)

            trade_ret = (pnl / total_buy_cost) if total_buy_cost > 0 else 0.0
            trade_returns.append(trade_ret)
    else:
        n_trades = 0

    if trade_returns:
        trade_returns_arr = np.array(trade_returns)
        winning_trades = trade_returns_arr[trade_returns_arr > 0]
        losing_trades = trade_returns_arr[trade_returns_arr <= 0]
        
        win_rate = len(winning_trades) / len(trade_returns_arr) if len(trade_returns_arr) > 0 else 0.0
        avg_trade_return = trade_returns_arr.mean() if len(trade_returns_arr) > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    # Profit Factor
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Average Holding Period
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0

    # 9. Turnover Rate
    trade_vol_col = "Value" if "Value" in trade_log_df.columns else None
    total_trade_volume = trade_log_df[trade_vol_col].sum() if (not trade_log_df.empty and trade_vol_col) else 0.0
    avg_portfolio_val = portfolio_values.mean() if len(portfolio_values) > 0 else initial_capital
    turnover = (total_trade_volume / (2.0 * avg_portfolio_val)) / years if (years > 0 and avg_portfolio_val > 0) else 0.0

    # 10. Exposure & Hit Ratio
    exposure = (portfolio_df["Shares"] > 0).mean() if "Shares" in portfolio_df.columns else 0.0
    hit_ratio = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Average Holding Period": avg_holding_period,
        "Average Trade Return": avg_trade_return,
        "Number of Trades": n_trades,
        "Turnover": turnover,
        "Exposure": exposure,
        "Hit Ratio": hit_ratio,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
    }


if __name__ == "__main__":
    print("Testing metrics.py for Phase 3...")
    try:
        from strategy import generate_signals
        from backtester import run_backtest

        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "RELIANCE.csv")
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path, index_col="Date")
            df_sample.index = pd.to_datetime(df_sample.index)
            
            df_sig = generate_signals(df_sample, strategy_name="SMA")
            port_df, trade_df = run_backtest(df_sig)
            metrics = calculate_metrics(port_df, trade_df)
            print("Calculated Metrics Test:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Self-test error: {e}")
