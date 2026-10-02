"""Position and P&L Engine for QuantLab Backtesting Engine."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .config import BacktestConfig


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """
    Executes the backtest simulation on normalized trade signals.
    Calculates quantities, allocated capital, gross and net P&L,
    cumulative returns, and discrepancy between provided and calculated returns.
    """
    trades = df.copy()

    # Sort chronologically by entry_date, then exit_date, then _raw_row_idx
    trades = trades.sort_values(
        by=["entry_date", "exit_date", "_raw_row_idx"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    # Determine trade direction based on strategy_mode or per-trade signal
    if config.strategy_mode == "long_only":
        trades["direction"] = "LONG"
    elif config.strategy_mode == "short_only":
        trades["direction"] = "SHORT"
    elif config.strategy_mode == "long_short":
        # Keep existing direction or default to LONG
        trades["direction"] = trades["direction"].fillna("LONG")

    n_trades = len(trades)
    trade_ids = [f"T{i+1:04d}" for i in range(n_trades)]
    trades["trade_id"] = trade_ids

    # Duration in days
    trades["duration"] = (trades["exit_date"] - trades["entry_date"]).dt.days
    trades["duration"] = trades["duration"].fillna(0).astype(int)

    # Position sizing & P&L calculation
    allocated_cap_list = []
    qty_list = []
    gross_pnl_list = []
    net_pnl_list = []
    calc_return_pct_list = []

    current_capital = config.initial_capital

    for _, row in trades.iterrows():
        if config.position_sizing == "fixed":
            alloc = config.initial_capital * config.capital_allocation_per_trade
        else:
            # compounding
            alloc = current_capital * config.capital_allocation_per_trade

        entry_p = float(row["entry_price"])
        exit_p = float(row["exit_price"])
        direction = row["direction"]

        qty = alloc / entry_p if entry_p > 0 else 0.0

        if direction == "LONG":
            gross_pnl = (exit_p - entry_p) * qty
            calc_ret = ((exit_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0
        else:
            # SHORT
            gross_pnl = (entry_p - exit_p) * qty
            calc_ret = ((entry_p - exit_p) / entry_p) * 100.0 if entry_p > 0 else 0.0

        # Cost and slippage
        cost = 0.0
        if config.transaction_cost > 0:
            cost += alloc * config.transaction_cost
        if config.slippage > 0:
            cost += alloc * config.slippage

        net_pnl = gross_pnl - cost
        current_capital += net_pnl

        allocated_cap_list.append(alloc)
        qty_list.append(qty)
        gross_pnl_list.append(gross_pnl)
        net_pnl_list.append(net_pnl)
        calc_return_pct_list.append(calc_ret)

    trades["allocated_capital"] = allocated_cap_list
    trades["quantity"] = qty_list
    trades["gross_pnl"] = gross_pnl_list
    trades["pnl"] = net_pnl_list
    trades["calc_return_pct"] = calc_return_pct_list

    # Reference return validation
    trades["provided_return_pct"] = trades["provided_return"]
    trades["return_diff_pct"] = trades["calc_return_pct"] - trades["provided_return_pct"]

    # Winning / Losing flags
    trades["winning_trade"] = trades["pnl"] > 0
    trades["losing_trade"] = trades["pnl"] < 0

    # Cumulative series
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    trades["cumulative_return_pct"] = (trades["cumulative_pnl"] / config.initial_capital) * 100.0
    trades["equity"] = config.initial_capital + trades["cumulative_pnl"]

    return trades


def generate_trade_log_excel(trades: pd.DataFrame, output_path: str) -> None:
    """Generate output/trade_log.xlsx with all requested columns formatted cleanly."""
    export_df = pd.DataFrame({
        "Trade ID": trades["trade_id"],
        "Stock": trades["stock"],
        "Entry Date": trades["entry_date"].dt.strftime("%Y-%m-%d"),
        "Exit Date": trades["exit_date"].dt.strftime("%Y-%m-%d"),
        "Duration": trades["duration"],
        "Direction": trades["direction"],
        "Entry Price": trades["entry_price"].round(2),
        "Exit Price": trades["exit_price"].round(2),
        "Quantity": trades["quantity"].round(4),
        "Allocated Capital": trades["allocated_capital"].round(2),
        "Gross P&L": trades["gross_pnl"].round(2),
        "Return %": trades["calc_return_pct"].round(4),
        "Cumulative P&L": trades["cumulative_pnl"].round(2),
        "Cumulative Return %": trades["cumulative_return_pct"].round(4),
        "Winning Trade": trades["winning_trade"],
        "Losing Trade": trades["losing_trade"]
    })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(output_path, index=False, sheet_name="Trade Log")
