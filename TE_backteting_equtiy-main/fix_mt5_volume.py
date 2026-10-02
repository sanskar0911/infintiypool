import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# INPUT / OUTPUT
# ---------------------------------------------------------

input_file = Path("data/metatrader5/RELIANCE_MT5.csv")
output_file = Path("data/metatrader5/RELIANCE_MT5_FIXED.csv")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("=" * 60)
print("QUANTLAB MT5 DATA FIX")
print("=" * 60)

print("Input rows:", len(df))

# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

df.columns = [
    col.strip().replace("<", "").replace(">", "")
    for col in df.columns
]

print("Columns:")
print(df.columns.tolist())

# ---------------------------------------------------------
# CONVERT NUMERIC COLUMNS
# ---------------------------------------------------------

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "TickVolume",
    "RealVolume",
    "Spread"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ---------------------------------------------------------
# REMOVE INVALID OHLC ROWS
# ---------------------------------------------------------

df = df.dropna(
    subset=[
        "DateTime",
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

# ---------------------------------------------------------
# FIX TICK VOLUME
# ---------------------------------------------------------
#
# The original file has TickVolume = 1 for every bar.
#
# For validation we need a positive, internally consistent
# bar volume. We use RealVolume as the source where available.
#
# TickVolume must be an integer.
# ---------------------------------------------------------

df["TickVolume"] = df["RealVolume"]

# Replace missing/zero values
df["TickVolume"] = df["TickVolume"].fillna(1)

df.loc[
    df["TickVolume"] <= 0,
    "TickVolume"
] = 1

# Convert to integer
df["TickVolume"] = (
    df["TickVolume"]
    .round()
    .astype("int64")
)

# ---------------------------------------------------------
# SPREAD
# ---------------------------------------------------------
#
# Keep spread at zero because this is a custom OHLC
# validation dataset and we are not validating spread.
# ---------------------------------------------------------

df["Spread"] = 0

# ---------------------------------------------------------
# REAL VOLUME
# ---------------------------------------------------------

df["RealVolume"] = (
    df["RealVolume"]
    .fillna(0)
    .round()
    .astype("int64")
)

# ---------------------------------------------------------
# FORMAT DATETIME
# ---------------------------------------------------------

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    errors="coerce"
)

df = df.dropna(
    subset=["DateTime"]
)

# MT5-style date format
df["DateTime"] = df["DateTime"].dt.strftime(
    "%Y.%m.%d %H:%M:%S"
)

# ---------------------------------------------------------
# FINAL COLUMN ORDER
# ---------------------------------------------------------

df = df[
    [
        "DateTime",
        "Open",
        "High",
        "Low",
        "Close",
        "TickVolume",
        "RealVolume",
        "Spread"
    ]
]

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

print()
print("=" * 60)
print("FIXED DATA")
print("=" * 60)

print("Rows:", len(df))

print()
print("First 5 rows:")
print(df.head().to_string(index=False))

print()
print("TickVolume:")
print(df["TickVolume"].describe())

print()
print("Output:")
print(output_file)

print()
print("=" * 60)
print("DONE")
print("=" * 60)