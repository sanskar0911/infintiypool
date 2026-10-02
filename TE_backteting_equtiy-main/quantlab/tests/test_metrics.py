"""Unit tests for performance metrics calculation."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import BacktestConfig
from src.metrics import calculate_metrics, calculate_skewness, calculate_streaks
from src.drawdown import build_daily_equity_curve


@pytest.fixture
def sample_trades():
    """Known test dataset where values are manually calculated."""
    dates = pd.date_range("2023-01-01", periods=5, freq="30D")
    data = {
        "stock": ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E"],
        "entry_date": dates,
        "exit_date": dates + pd.Timedelta(days=15),
        "duration": [15, 15, 15, 15, 15],
        "direction": ["LONG"] * 5,
        "entry_price": [100.0, 200.0, 150.0, 250.0, 100.0],
        "exit_price": [110.0, 190.0, 165.0, 225.0, 120.0],  # returns: +10%, -5%, +10%, -10%, +20%
        "provided_return": [10.0, -5.0, 10.0, -10.0, 20.0],
        "_raw_row_idx": [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    # Allocation: fixed 2% of 10,000,000 = 200,000 per trade
    # Trade 1: +10% -> PnL = +20,000
    # Trade 2: -5%  -> PnL = -10,000
    # Trade 3: +10% -> PnL = +20,000
    # Trade 4: -10% -> PnL = -20,000
    # Trade 5: +20% -> PnL = +40,000
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
    df["trade_id"] = [f"T{i+1:04d}" for i in range(5)]
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["cumulative_return_pct"] = df["cumulative_pnl"] / 10000000.0 * 100.0
    df["equity"] = 10000000.0 + df["cumulative_pnl"]
    return df


def test_basic_trade_metrics(sample_trades):
    config = BacktestConfig(initial_capital=10000000.0, capital_allocation_per_trade=0.02)
    daily_df = build_daily_equity_curve(sample_trades, config)
    metrics = calculate_metrics(sample_trades, daily_df, config)

    # Total trades = 5
    assert metrics.total_trades == 5
    # Winners: T1, T3, T5 = 3
    assert metrics.winning_trades == 3
    # Losers: T2, T4 = 2
    assert metrics.losing_trades == 2
    # Win rate = 3 / 5 = 60%
    assert pytest.approx(metrics.win_rate, 1e-4) == 0.60
    # Loss rate = 2 / 5 = 40%
    assert pytest.approx(metrics.loss_rate, 1e-4) == 0.40

    # P&L sums:
    # Gross profit: 20000 + 20000 + 40000 = 80000
    assert pytest.approx(metrics.gross_profit, 1e-2) == 80000.0
    # Gross loss: -10000 + -20000 = -30000
    assert pytest.approx(metrics.gross_loss, 1e-2) == -30000.0
    # Total PnL: 80000 - 30000 = 50000
    assert pytest.approx(metrics.total_pnl, 1e-2) == 50000.0
    # Final equity = 10050000
    assert pytest.approx(metrics.final_equity, 1e-2) == 10050000.0

    # Profit factor: 80000 / 30000 = 2.6667
    assert pytest.approx(metrics.profit_factor, 1e-4) == 80000.0 / 30000.0

    # Average win: 80000 / 3 = 26666.67
    assert pytest.approx(metrics.avg_win, 1e-2) == 80000.0 / 3
    # Average loss: -30000 / 2 = -15000.0 (negative)
    assert pytest.approx(metrics.avg_loss, 1e-2) == -15000.0

    # Expectancy: (0.60 * 26666.6667) + (0.40 * -15000) = 16000 - 6000 = 10000
    assert pytest.approx(metrics.expectancy, 1e-2) == 10000.0


def test_skewness_formula():
    """Verify that sample-adjusted skewness formula matches scipy.stats.skew(bias=False)."""
    pnl = pd.Series([1000.0, -500.0, 2000.0, 1500.0, -300.0, 5000.0])
    my_skew = calculate_skewness(pnl)
    scipy_skew = float(stats.skew(pnl, bias=False))
    assert pytest.approx(my_skew, 1e-6) == scipy_skew


def test_streaks_calculation():
    """Verify max win and loss streaks."""
    pnl = pd.Series([100, 200, 300, -50, -60, 100, -20, -30, -40, -50, 100])
    win_s, loss_s, max_p, max_l = calculate_streaks(pnl)
    assert win_s == 3  # [100, 200, 300]
    assert loss_s == 4  # [-20, -30, -40, -50]
    assert max_p == 600.0
    assert max_l == -140.0
