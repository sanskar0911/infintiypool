"""
utils.py

Small shared helper functions used across the pipeline.
"""

import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load a raw OHLCV CSV and parse its Date column as the index."""
    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def daily_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.Series:
    """
    Compute simple daily percentage returns from a price column.

    Uses Adjusted Close by default, since raw Close will show fake jumps
    around stock splits / dividends (see Day 1 notes on Adjusted Close).
    """
    return df[price_col].pct_change()


def moving_average(df: pd.DataFrame, window: int, price_col: str = "Adj Close") -> pd.Series:
    """Compute a simple rolling moving average over `window` trading days."""
    return df[price_col].rolling(window=window).mean()


def summarize_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    return df.isnull().sum()
