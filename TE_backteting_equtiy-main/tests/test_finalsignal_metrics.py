"""
test_finalsignal_metrics.py

Unit and integration test verifying that the backtesting engine processes finalsignal.csv
and produces the authoritative backtest_metrics.csv with exact metrics, cash conservation,
and risk parameters as required by the teammate specification.
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation_signal.backtester1 import run_validation_simulation, export_backtest_metrics_csv


def test_finalsignal_backtest_metrics():
    # Locate finalsignal.csv
    candidates = [
        BASE_DIR / "src" / "validation_signal" / "finalsignal.csv",
        BASE_DIR / "data" / "finalsignal.csv",
        BASE_DIR / "finalsignal.csv",
        BASE_DIR.parent / "finalsignal.csv"
    ]
    sig_path = next((p for p in candidates if p.exists()), None)
    assert sig_path is not None, "finalsignal.csv must exist in workspace"

    out_dir = BASE_DIR / "results" / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_validation_simulation(
        file_path=str(sig_path),
        initial_capital=10000000.0,
        allocation=0.05,
        risk_free=0.06,
        strategy_mode="long_only",
        output_dir=str(out_dir)
    )

    assert res["status"] == "SUCCESS"
    assert res["trade_count"] == 97

    metrics = res["metrics"]

    # 1. Trade counts and hit rates
    assert metrics["Total Trades"] == 97
    assert metrics["Winning Trades"] == 45
    assert metrics["Losing Trades"] == 52
    assert metrics["Breakeven Trades"] == 0
    assert abs(metrics["Win Rate Raw"] - 46.3918) < 0.001
    assert abs(metrics["Loss Rate Raw"] - 53.6082) < 0.001

    # 2. Capital and Net P&L (Strict Cash Conservation)
    assert metrics["Initial Capital"] == 10000000.0
    assert abs(metrics["Total P&L Raw"] - 10294.5351) < 0.01
    assert abs(metrics["Final Equity Raw"] - 10010294.5351) < 0.01
    assert abs(metrics["Realized P&L Raw"] - 10294.5351) < 0.01
    assert abs(metrics["Total Return Raw"] - 0.1029) < 0.001
    assert abs(metrics["CAGR Raw"] - 0.0177) < 0.001

    # 3. Trade distribution & Profit factor
    assert abs(metrics["Average P&L Raw"] - 106.1292) < 0.01
    assert abs(metrics["Average Win Raw"] - 52201.2242) < 0.01
    assert abs(metrics["Average Loss Raw"] - (-44976.1645)) < 0.01
    assert abs(metrics["Profit Factor Raw"] - 1.0044) < 0.001
    assert abs(metrics["Risk/Reward Raw"] - 1.1606) < 0.001
    assert abs(metrics["Best Trade Raw"] - 201911.8538) < 0.01
    assert abs(metrics["Worst Trade Raw"] - (-156290.1393)) < 0.01
    assert abs(metrics["Median P&L Raw"] - (-7522.7644)) < 0.01
    assert abs(metrics["Average Holding Period Raw"] - 67.9485) < 0.01

    # 4. Streaks
    assert metrics["Max Win Streak"] == 7
    assert metrics["Max Loss Streak"] == 11

    # 5. Risk metrics & Realized Drawdown
    assert abs(metrics["Sharpe Ratio Raw"] - (-0.1575)) < 0.001
    assert abs(metrics["Sortino Ratio Raw"] - (-0.2202)) < 0.001
    assert abs(metrics["Calmar Ratio Raw"] - 0.0017) < 0.001
    assert abs(metrics["Maximum Drawdown % Raw"] - (-10.2154)) < 0.001
    assert abs(metrics["Maximum Drawdown (INR) Raw"] - 1096142.4402) < 0.01
    assert metrics["Max DD Duration (Days)"] == 1106
    assert metrics["Drawdown Peak Date"] == "2017-10-19"
    assert metrics["Drawdown Trough Date"] == "2020-03-20"
    assert abs(metrics["Skewness Raw"] - 0.3786) < 0.001

    # 6. Verify backtest_metrics.csv exists and has 49 metric rows
    csv_file = out_dir / "backtest_metrics.csv"
    assert csv_file.exists()
    lines = [line.strip() for line in open(csv_file, encoding="utf-8") if line.strip()]
    assert len(lines) == 50  # 1 header + 49 metrics
    assert lines[0] == "Metric,Display Value,Raw Value"
    assert lines[1] == '"Total Return","0.10%","0.1029"'
    assert lines[3] == '"Total P&L","₹10,294.54","10294.5351"'
    assert lines[4] == '"Annualized Volatility","NaN","N/A"'
    assert lines[5] == '"Sharpe Ratio","-0.158","-0.1575"'
    assert lines[6] == '"Sortino Ratio","-0.220","-0.2202"'
