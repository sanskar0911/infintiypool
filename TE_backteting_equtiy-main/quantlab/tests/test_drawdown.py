"""Unit tests for drawdown calculation and equity series."""

import numpy as np
import pandas as pd
import pytest

from src.config import BacktestConfig
from src.drawdown import build_daily_equity_curve, calculate_drawdown_metrics


def test_drawdown_calculation():
    # Construct synthetic daily equity series
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Equity progression: 100 -> 120 -> 110 -> 90 -> 100 -> 125 -> 120 -> 130 -> 110 -> 130
    equities = [100.0, 120.0, 110.0, 90.0, 100.0, 125.0, 120.0, 130.0, 110.0, 130.0]
    peaks =    [100.0, 120.0, 120.0, 120.0, 120.0, 125.0, 125.0, 130.0, 130.0, 130.0]
    dds =      [0.0,   0.0,   -10.0, -30.0, -20.0, 0.0,   -5.0,  0.0,   -20.0, 0.0]

    daily_df = pd.DataFrame({
        "date": dates,
        "equity": equities,
        "running_peak": peaks,
        "drawdown": dds,
        "drawdown_pct": [(d / p) * 100.0 for d, p in zip(dds, peaks)]
    })

    metrics = calculate_drawdown_metrics(daily_df)

    # Worst drawdown was -30.0 at index 3 (2023-01-04)
    assert metrics.max_drawdown == -30.0
    # Max DD % = -30 / 120 = -25%
    assert pytest.approx(metrics.max_drawdown_pct, 1e-4) == -25.0
    # Peak date before trough is index 1 (2023-01-02)
    assert metrics.peak_date == "2023-01-02"
    # Trough date is index 3 (2023-01-04)
    assert metrics.trough_date == "2023-01-04"
    # Recovery date: reached 125 on index 5 (2023-01-06)
    assert metrics.recovery_date == "2023-01-06"
    # Duration: (2023-01-06 - 2023-01-02) = 4 days
    assert metrics.max_drawdown_duration == 4
