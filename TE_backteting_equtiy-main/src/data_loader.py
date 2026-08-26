"""
data_loader.py

Ingests and loads historical daily OHLCV data for NSE stocks.
"""

import os
import time
import pandas as pd

TICKERS = {
    "RELIANCE.NS": "RELIANCE",
    "TCS.NS": "TCS",
    "INFY.NS": "INFY",
    "HDFCBANK.NS": "HDFCBANK",
    "ICICIBANK.NS": "ICICIBANK",
}

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock OHLCV data for given ticker from processed or raw data folder.
    """
    processed_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}.csv")
    raw_path = os.path.join(BASE_DIR, "data", "raw", f"{ticker}.csv")

    target_path = processed_path if os.path.exists(processed_path) else raw_path

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Price data file for {ticker} not found at {target_path}")

    df = pd.read_csv(target_path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def download_stock(yahoo_ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker via yfinance."""
    import yfinance as yf
    df = yf.download(
        yahoo_ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {yahoo_ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def download_all(tickers: dict, start: str, end: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for yahoo_ticker, file_name in tickers.items():
        print(f"Downloading {yahoo_ticker} ...")
        try:
            df = download_stock(yahoo_ticker, start, end)
            out_path = os.path.join(output_dir, f"{file_name}.csv")
            df.to_csv(out_path)
            print(f"  Saved {len(df)} rows -> {out_path}")
        except Exception as e:
            print(f"  FAILED for {yahoo_ticker}: {e}")
        time.sleep(1)


if __name__ == "__main__":
    download_all(TICKERS, START_DATE, END_DATE, OUTPUT_DIR)
