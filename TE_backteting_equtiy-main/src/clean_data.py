"""
clean_data.py

Reads raw OHLCV CSVs from data/raw/, cleans them, and writes the cleaned
versions to data/processed/.

Cleaning steps (Day 4):
1. Parse the Date column properly as a datetime index.
2. Report and handle missing values.
3. Drop duplicate rows.
4. Ensure correct datatypes for all OHLCV columns.
5. Sort by date ascending (critical — an unsorted date index is a common,
   silent source of look-ahead bias in rolling calculations like
   .rolling() or .pct_change()).
6. Add a few standard derived columns useful for downstream analysis:
   daily returns and a 50-day moving average.

Run:
    python src/clean_data.py
"""

import os
import glob
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def clean_single_file(path: str) -> pd.DataFrame:
    ticker = os.path.splitext(os.path.basename(path))[0]
    print(f"\nCleaning {ticker} ...")

    # --- Load & parse dates ---
    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # --- Check missing values (report before fixing) ---
    missing_before = df.isnull().sum()
    total_missing = missing_before.sum()
    if total_missing > 0:
        print(f"  Missing values found:\n{missing_before[missing_before > 0]}")
    else:
        print("  No missing values found.")

    # --- Check duplicate rows/dates ---
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        print(f"  Found {n_dupes} duplicate date rows — dropping, keeping first.")
        df = df[~df.index.duplicated(keep="first")]
    else:
        print("  No duplicate dates found.")

    # --- Ensure correct datatypes ---
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Handle any missing values introduced by type coercion ---
    # Forward-fill price gaps (assume market was simply closed / data feed
    # hiccup), but never fill Volume with a fabricated non-zero value.
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    # Drop any remaining rows that are still fully empty (e.g. leading NaNs
    # before the first valid trading day).
    df = df.dropna(how="all", subset=price_cols)

    # --- Sort chronologically — critical for correct rolling calculations ---
    df = df.sort_index()

    # --- Derived columns useful for EDA / backtesting ---
    if "Adj Close" in df.columns:
        df["Returns"] = df["Adj Close"].pct_change()
        df["MA50"] = df["Adj Close"].rolling(window=50).mean()

    print(f"  Final shape: {df.shape[0]} rows x {df.shape[1]} columns "
          f"({df.index.min().date()} -> {df.index.max().date()})")

    return df


def clean_all() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    if not raw_files:
        print(f"No raw CSV files found in {RAW_DIR}. "
              f"Run src/data_loader.py first.")
        return

    for path in raw_files:
        ticker = os.path.splitext(os.path.basename(path))[0]
        cleaned = clean_single_file(path)
        out_path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        cleaned.to_csv(out_path)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    clean_all()
