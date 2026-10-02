import pandas as pd
from pathlib import Path

# ============================================================
# QUANTLAB - MT5 vs Python SMA Validation
# ============================================================

# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path("data") / "metatrader5" / "RELIANCE_MT5.csv"
OUTPUT_FILE = Path("results") / "mt5_python_sma_validation.csv"

# ------------------------------------------------------------
# STRATEGY PARAMETERS
# ------------------------------------------------------------

FAST_PERIOD = 20
SLOW_PERIOD = 50


print("=" * 60)
print("QUANTLAB SMA 20/50 VALIDATION")
print("=" * 60)


# ============================================================
# 1. CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE.resolve()}"
    )


# ============================================================
# 2. LOAD MT5 CSV
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded file: {INPUT_FILE}")
print(f"Rows loaded: {len(df)}")


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = [
    col.strip("<>").strip()
    for col in df.columns
]

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "DateTime",
    "Open",
    "High",
    "Low",
    "Close"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. CONVERT DATETIME
# ============================================================

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    format="%Y.%m.%d %H:%M:%S",
    errors="coerce"
)

# Check for invalid dates

invalid_dates = df["DateTime"].isna().sum()

if invalid_dates > 0:
    print(
        f"\nWARNING: {invalid_dates} invalid datetime values found."
    )

    df = df.dropna(
        subset=["DateTime"]
    )


# ============================================================
# 6. SORT CHRONOLOGICALLY
# ============================================================

df = (
    df
    .sort_values("DateTime")
    .reset_index(drop=True)
)


# ============================================================
# 7. CONVERT PRICE COLUMN
# ============================================================

df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

# Remove rows without valid Close prices

df = df.dropna(
    subset=["Close"]
).copy()


# ============================================================
# 8. CALCULATE SMA 20
# ============================================================

df["SMA20"] = (
    df["Close"]
    .rolling(
        window=FAST_PERIOD,
        min_periods=FAST_PERIOD
    )
    .mean()
)


# ============================================================
# 9. CALCULATE SMA 50
# ============================================================

df["SMA50"] = (
    df["Close"]
    .rolling(
        window=SLOW_PERIOD,
        min_periods=SLOW_PERIOD
    )
    .mean()
)


# ============================================================
# 10. PREVIOUS SMA VALUES
# ============================================================

df["SMA20_Previous"] = df["SMA20"].shift(1)

df["SMA50_Previous"] = df["SMA50"].shift(1)


# ============================================================
# 11. BUY SIGNAL
#
# Previous candle:
# SMA20 <= SMA50
#
# Current candle:
# SMA20 > SMA50
# ============================================================

df["BUY"] = (
    (df["SMA20_Previous"] <= df["SMA50_Previous"])
    &
    (df["SMA20"] > df["SMA50"])
)


# ============================================================
# 12. SELL / EXIT SIGNAL
#
# Previous candle:
# SMA20 >= SMA50
#
# Current candle:
# SMA20 < SMA50
# ============================================================

df["SELL"] = (
    (df["SMA20_Previous"] >= df["SMA50_Previous"])
    &
    (df["SMA20"] < df["SMA50"])
)


# ============================================================
# 13. GET SIGNAL DATA
# ============================================================

buy_signals = df[
    df["BUY"]
].copy()

sell_signals = df[
    df["SELL"]
].copy()


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Total rows       : {len(df)}")
print(f"SMA20 period     : {FAST_PERIOD}")
print(f"SMA50 period     : {SLOW_PERIOD}")
print(f"BUY signals      : {len(buy_signals)}")
print(f"SELL signals     : {len(sell_signals)}")


# ============================================================
# 15. PRINT BUY SIGNALS
# ============================================================

print("\n" + "=" * 60)
print("BUY SIGNALS")
print("=" * 60)

if buy_signals.empty:

    print("No BUY signals found.")

else:

    print(
        buy_signals[
            [
                "DateTime",
                "Close",
                "SMA20",
                "SMA50"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 16. PRINT SELL SIGNALS
# ============================================================

print("\n" + "=" * 60)
print("SELL / EXIT SIGNALS")
print("=" * 60)

if sell_signals.empty:

    print("No SELL signals found.")

else:

    print(
        sell_signals[
            [
                "DateTime",
                "Close",
                "SMA20",
                "SMA50"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 17. CREATE RESULTS DIRECTORY
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 18. SAVE VALIDATION RESULTS
# ============================================================

output_columns = [
    "DateTime",
    "Open",
    "High",
    "Low",
    "Close",
    "SMA20",
    "SMA50",
    "BUY",
    "SELL"
]

df[output_columns].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION FILE SAVED")
print("=" * 60)

print(
    f"Output: {OUTPUT_FILE}"
)

print("\nValidation completed successfully.")
print("=" * 60)