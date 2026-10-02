"""Unit tests for monthly and yearly return logs."""

import pandas as pd
import pytest

from src.returns import calculate_monthly_and_yearly_returns


def test_monthly_and_yearly_returns():
    # 2 years of daily data
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    # Linear growth in 2022 from 100 to 110 (+10%), in 2023 from 110 to 132 (+20%)
    equity = []
    for d in dates:
        if d.year == 2022:
            progress = (d - pd.Timestamp("2022-01-01")).days / 364.0
            equity.append(100.0 + progress * 10.0)
        else:
            progress = (d - pd.Timestamp("2023-01-01")).days / 364.0
            equity.append(110.0 + progress * 22.0)

    daily_df = pd.DataFrame({"date": dates, "equity": equity})
    monthly_df, yearly_df = calculate_monthly_and_yearly_returns(daily_df)

    assert len(yearly_df) == 2
    assert yearly_df.loc[0, "Year"] == 2022
    assert yearly_df.loc[1, "Year"] == 2023
    assert pytest.approx(yearly_df.loc[0, "Return %"], 0.2) == 10.0
    assert pytest.approx(yearly_df.loc[1, "Return %"], 0.2) == 20.0

    assert len(monthly_df) == 2
    assert "Jan" in monthly_df.columns
    assert "Dec" in monthly_df.columns
    assert "Annual Return %" in monthly_df.columns
