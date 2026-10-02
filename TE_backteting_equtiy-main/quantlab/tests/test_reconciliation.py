"""Unit tests for reconciliation between Python engine and Excel calculations."""

import pandas as pd
import pytest

from src.config import BacktestConfig
from src.metrics import calculate_metrics
from src.drawdown import build_daily_equity_curve
from src.reconciliation import reconcile_metrics


def test_reconciliation_exact_match():
    dates = pd.date_range("2023-01-01", periods=10, freq="15D")
    df = pd.DataFrame({
        "stock": [f"STOCK_{i}" for i in range(10)],
        "entry_date": dates,
        "exit_date": dates + pd.Timedelta(days=10),
        "duration": [10] * 10,
        "direction": ["LONG"] * 10,
        "entry_price": [100.0] * 10,
        "exit_price": [110.0, 90.0, 105.0, 95.0, 120.0, 80.0, 110.0, 100.0, 115.0, 95.0],
        "provided_return": [10.0, -10.0, 5.0, -5.0, 20.0, -20.0, 10.0, 0.0, 15.0, -5.0],
        "_raw_row_idx": list(range(1, 11))
    })
    alloc = 200000.0
    df["allocated_capital"] = alloc
    df["quantity"] = alloc / df["entry_price"]
    df["gross_pnl"] = (df["exit_price"] - df["entry_price"]) * df["quantity"]
    df["pnl"] = df["gross_pnl"]
    df["calc_return_pct"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100.0
    df["provided_return_pct"] = df["provided_return"]
    df["return_diff_pct"] = 0.0
    df["winning_trade"] = df["pnl"] > 0
    df["losing_trade"] = df["pnl"] < 0
    df["trade_id"] = [f"T{i+1:04d}" for i in range(10)]
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["cumulative_return_pct"] = df["cumulative_pnl"] / 10000000.0 * 100.0
    df["equity"] = 10000000.0 + df["cumulative_pnl"]

    config = BacktestConfig(initial_capital=10000000.0, capital_allocation_per_trade=0.02, tolerance=0.0001)
    daily_df = build_daily_equity_curve(df, config)
    metrics = calculate_metrics(df, daily_df, config)

    rec_df, matches, total = reconcile_metrics(metrics, config)

    assert total > 0
    assert matches == total
    assert (rec_df["Status"] == "MATCH").all()
