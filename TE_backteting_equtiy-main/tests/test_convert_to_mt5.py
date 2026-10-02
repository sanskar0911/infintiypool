"""
tests/test_convert_to_mt5.py

Unit tests for MT5 custom symbol bar conversion utility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tools.convert_to_mt5 import (
    ConversionValidationError,
    IntegrityCheckError,
    convert_to_mt5,
    detect_columns,
    run_integrity_check,
    save_mt5_csv,
    validate_source_data,
)


@pytest.fixture
def sample_ohlcv_df():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Open": [100.0, 102.0, 101.0],
        "High": [105.0, 106.0, 104.0],
        "Low": [99.0, 101.0, 98.0],
        "Close": [103.0, 104.0, 102.0],
        "Adj Close": [98.0, 99.0, 97.0],
        "Volume": [1000, 2000, 1500],
        "Returns": [np.nan, 0.0097, -0.019],
        "MA50": [np.nan, np.nan, np.nan],
    })


def test_detect_columns(sample_ohlcv_df):
    col_map = detect_columns(list(sample_ohlcv_df.columns))
    assert col_map["date"] == "Date"
    assert col_map["open"] == "Open"
    assert col_map["high"] == "High"
    assert col_map["low"] == "Low"
    assert col_map["close"] == "Close"
    assert col_map["volume"] == "Volume"


def test_detect_columns_missing_required():
    bad_columns = ["Date", "Open", "High"]
    with pytest.raises(ConversionValidationError) as exc:
        detect_columns(bad_columns)
    assert "Missing required OHLC field" in str(exc.value)


def test_validate_source_data_success(sample_ohlcv_df):
    col_map = detect_columns(list(sample_ohlcv_df.columns))
    stats = validate_source_data(sample_ohlcv_df, col_map)
    assert stats["input_rows"] == 3
    assert stats["first_date"] == "2024-01-01"
    assert stats["last_date"] == "2024-01-03"
    assert stats["num_invalid_rows"] == 0


def test_validate_source_data_inverted_high_low():
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Open": [100.0],
        "High": [90.0],  # Invalid: High < Open
        "Low": [80.0],
        "Close": [85.0],
        "Volume": [1000],
    })
    col_map = detect_columns(list(df.columns))
    with pytest.raises(ConversionValidationError) as exc:
        validate_source_data(df, col_map)
    assert "High >= max(Open, Close, Low)" in str(exc.value)


def test_convert_to_mt5_constant(sample_ohlcv_df):
    col_map = detect_columns(list(sample_ohlcv_df.columns))
    mt5_df = convert_to_mt5(sample_ohlcv_df, col_map, tick_volume_mode="constant", spread=5)

    expected_cols = [
        "<DateTime>", "<Open>", "<High>", "<Low>", "<Close>", "<TickVolume>", "<RealVolume>", "<Spread>"
    ]
    assert list(mt5_df.columns) == expected_cols
    assert mt5_df["<DateTime>"].iloc[0] == "2024.01.01 00:00:00"
    assert mt5_df["<TickVolume>"].iloc[0] == 1
    assert mt5_df["<RealVolume>"].iloc[0] == 1000
    assert mt5_df["<Spread>"].iloc[0] == 5


def test_convert_to_mt5_volume_mode(sample_ohlcv_df):
    col_map = detect_columns(list(sample_ohlcv_df.columns))
    mt5_df = convert_to_mt5(sample_ohlcv_df, col_map, tick_volume_mode="volume", spread=0)
    assert mt5_df["<TickVolume>"].iloc[0] == 1000
    assert mt5_df["<RealVolume>"].iloc[0] == 1000


def test_roundtrip_integrity_check(tmp_path, sample_ohlcv_df):
    col_map = detect_columns(list(sample_ohlcv_df.columns))
    validate_source_data(sample_ohlcv_df, col_map)
    mt5_df = convert_to_mt5(sample_ohlcv_df, col_map, tick_volume_mode="constant", spread=0)

    out_file = tmp_path / "TEST_MT5.csv"
    save_mt5_csv(mt5_df, out_file)

    assert run_integrity_check(sample_ohlcv_df, col_map, out_file) is True
