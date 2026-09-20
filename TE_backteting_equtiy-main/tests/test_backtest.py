"""
test_backtest.py

Unit test suite for signals, backtesting simulation engine, metrics, and benchmark comparison.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.strategy import generate_signals
from src.backtester import run_backtest
from src.metrics import calculate_metrics
from src.benchmark import compare_strategy_vs_benchmark, load_benchmark_data


@pytest.fixture
def sample_price_data():
    """Create a synthetic 100-day price DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": 1000
    }, index=dates)
    return df


def test_generate_signals(sample_price_data):
    """Test SMA signal generation."""
    df_sig = generate_signals(sample_price_data, strategy_name="SMA", short_window=5, long_window=20)
    assert "Signal" in df_sig.columns
    assert set(df_sig["Signal"].unique()).issubset({-1, 0, 1})


def test_run_backtest(sample_price_data):
    """Test backtesting engine simulation."""
    df_sig = generate_signals(sample_price_data, strategy_name="SMA", short_window=5, long_window=20)
    portfolio_df, trade_log_df = run_backtest(df_sig, initial_capital=10000.0)

    assert "Portfolio Value" in portfolio_df.columns
    assert len(portfolio_df) == len(sample_price_data)
    assert portfolio_df["Portfolio Value"].iloc[0] == 10000.0


def test_calculate_metrics(sample_price_data):
    """Test deterministic metrics calculation."""
    df_sig = generate_signals(sample_price_data, strategy_name="SMA", short_window=5, long_window=20)
    portfolio_df, trade_log_df = run_backtest(df_sig, initial_capital=10000.0)
    metrics = calculate_metrics(portfolio_df, trade_log_df, initial_capital=10000.0)

    assert "Total Return" in metrics
    assert "CAGR" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Maximum Drawdown" in metrics
    assert isinstance(metrics["Sharpe Ratio"], float)


def test_benchmark_comparison(sample_price_data):
    """Test strategy vs benchmark comparison function."""
    df_sig = generate_signals(sample_price_data, strategy_name="SMA", short_window=5, long_window=20)
    portfolio_df, trade_log_df = run_backtest(df_sig, initial_capital=10000.0)

    res = compare_strategy_vs_benchmark(portfolio_df, sample_price_data, benchmark_symbol="NIFTY50", initial_capital=10000.0)
    assert res["status"] == "SUCCESS"
    assert "strategy_cagr" in res
    assert "benchmark_cagr" in res