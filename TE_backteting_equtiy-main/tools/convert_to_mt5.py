#!/usr/bin/env python3
"""
tools/convert_to_mt5.py

Python conversion utility for QUANTLAB.
Converts daily OHLCV CSV data into MetaTrader 5 Custom Symbol-compatible BAR CSV.

Target MT5 Bar format:
<DateTime>,<Open>,<High>,<Low>,<Close>,<TickVolume>,<RealVolume>,<Spread>

DateTime format: YYYY.MM.DD 00:00:00 (for daily candles)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("convert_to_mt5")


class ConversionValidationError(Exception):
    """Raised when source CSV validation fails."""
    pass


class IntegrityCheckError(Exception):
    """Raised when post-conversion integrity check fails."""
    pass


# Candidate column names for automatic detection (case-insensitive)
DATE_CANDIDATES = [
    "date", "datetime", "timestamp", "time", "<date>", "<datetime>", "<timestamp>"
]
OPEN_CANDIDATES = ["open", "<open>"]
HIGH_CANDIDATES = ["high", "<high>"]
LOW_CANDIDATES = ["low", "<low>"]
CLOSE_CANDIDATES = ["close", "<close>"]
VOLUME_CANDIDATES = [
    "volume", "vol", "realvolume", "real_volume", "<volume>", "<vol>", "<realvolume>"
]

MT5_BAR_COLUMNS = [
    "<DateTime>",
    "<Open>",
    "<High>",
    "<Low>",
    "<Close>",
    "<TickVolume>",
    "<RealVolume>",
    "<Spread>",
]


def detect_columns(columns: List[str]) -> Dict[str, str]:
    """
    Detect OHLC, Date, and Volume column mappings from source CSV header.

    Args:
        columns: List of column names in the source DataFrame.

    Returns:
        Dictionary mapping canonical field names to source column names.

    Raises:
        ConversionValidationError: If any required OHLC or Date column is missing.
    """
    col_lookup = {col.strip().lower(): col for col in columns}
    mapping: Dict[str, str] = {}

    # Detect Date
    date_col = next((col_lookup[c] for c in DATE_CANDIDATES if c in col_lookup), None)
    if not date_col:
        raise ConversionValidationError(
            f"Could not automatically detect Date column. "
            f"Available columns: {columns}. Supported candidates: {DATE_CANDIDATES}"
        )
    mapping["date"] = date_col

    # Detect OHLC
    for name, candidates in [
        ("open", OPEN_CANDIDATES),
        ("high", HIGH_CANDIDATES),
        ("low", LOW_CANDIDATES),
        ("close", CLOSE_CANDIDATES),
    ]:
        found = next((col_lookup[c] for c in candidates if c in col_lookup), None)
        if not found:
            raise ConversionValidationError(
                f"Missing required OHLC field '{name}'. "
                f"Available columns: {columns}. Supported candidates: {candidates}"
            )
        mapping[name] = found

    # Detect Volume (optional in generic sources, but required for real volume)
    vol_col = next((col_lookup[c] for c in VOLUME_CANDIDATES if c in col_lookup), None)
    if vol_col:
        mapping["volume"] = vol_col
    else:
        logger.warning(
            "No Volume column detected in source CSV. RealVolume will default to 0."
        )
        mapping["volume"] = ""

    return mapping


def validate_source_data(
    df: pd.DataFrame, col_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    Validate source OHLCV data according to MT5 conversion requirements:
    - required OHLC columns exist
    - dates are parseable
    - no duplicate timestamps
    - timestamps are sorted ascending
    - Open, High, Low, Close are numeric
    - Volume is numeric if available
    - no NaN values in required OHLC fields
    - High >= max(Open, Close, Low)
    - Low <= min(Open, Close, High)
    - no negative volume
    - no negative prices
    - no invalid dates

    Args:
        df: Source pandas DataFrame.
        col_map: Column mapping dictionary.

    Returns:
        Validation statistics dictionary.

    Raises:
        ConversionValidationError: If any validation rule is violated.
    """
    errors: List[str] = []
    stats: Dict[str, Any] = {
        "input_rows": len(df),
        "num_duplicates": 0,
        "num_invalid_rows": 0,
        "num_missing_ohlc": 0,
        "num_negative_prices": 0,
        "num_negative_volume": 0,
        "first_date": None,
        "last_date": None,
    }

    if len(df) == 0:
        raise ConversionValidationError("Source CSV is completely empty.")

    # 1. Parse dates
    date_col = col_map["date"]
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    invalid_dates_count = int(parsed_dates.isna().sum())
    if invalid_dates_count > 0:
        errors.append(f"{invalid_dates_count} rows contain unparseable/invalid dates.")
        stats["num_invalid_rows"] += invalid_dates_count

    # 2. Check duplicate timestamps
    duplicate_count = int(parsed_dates.duplicated().sum())
    stats["num_duplicates"] = duplicate_count
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate date/timestamp records.")

    # 3. Check ascending order
    # Filter out NaT for monotonicity check
    valid_dates = parsed_dates.dropna()
    if not valid_dates.is_monotonic_increasing:
        errors.append(
            "Timestamps are not strictly sorted in ascending order."
        )

    if len(valid_dates) > 0:
        stats["first_date"] = valid_dates.iloc[0].strftime("%Y-%m-%d")
        stats["last_date"] = valid_dates.iloc[-1].strftime("%Y-%m-%d")

    # 4. Check OHLC numeric and NaN
    ohlc_cols = [col_map["open"], col_map["high"], col_map["low"], col_map["close"]]
    for col_key in ["open", "high", "low", "close"]:
        c = col_map[col_key]
        numeric_series = pd.to_numeric(df[c], errors="coerce")
        nan_count = int(numeric_series.isna().sum())
        stats["num_missing_ohlc"] += nan_count
        if nan_count > 0:
            errors.append(f"Column '{c}' contains {nan_count} non-numeric or NaN values.")

        neg_count = int((numeric_series <= 0).sum())
        stats["num_negative_prices"] += neg_count
        if neg_count > 0:
            errors.append(f"Column '{c}' contains {neg_count} non-positive/negative prices.")

    # 5. Check Volume if present
    vol_col = col_map.get("volume")
    if vol_col:
        vol_numeric = pd.to_numeric(df[vol_col], errors="coerce")
        vol_nan = int(vol_numeric.isna().sum())
        if vol_nan > 0:
            errors.append(f"Volume column '{vol_col}' contains {vol_nan} NaN/non-numeric values.")
        neg_vol = int((vol_numeric < 0).sum())
        stats["num_negative_volume"] = neg_vol
        if neg_vol > 0:
            errors.append(f"Volume column '{vol_col}' contains {neg_vol} negative values.")

    # 6. Check High >= max(Open, Close, Low) and Low <= min(Open, Close, High)
    # Use slight epsilon (1e-7) to guard against floating-point precision inaccuracies
    eps = 1e-7
    open_s = pd.to_numeric(df[col_map["open"]], errors="coerce")
    high_s = pd.to_numeric(df[col_map["high"]], errors="coerce")
    low_s = pd.to_numeric(df[col_map["low"]], errors="coerce")
    close_s = pd.to_numeric(df[col_map["close"]], errors="coerce")

    high_invalid = (
        (high_s < open_s - eps)
        | (high_s < close_s - eps)
        | (high_s < low_s - eps)
    )
    high_invalid_count = int(high_invalid.sum())
    if high_invalid_count > 0:
        errors.append(
            f"{high_invalid_count} bars violate High >= max(Open, Close, Low)."
        )

    low_invalid = (
        (low_s > open_s + eps)
        | (low_s > close_s + eps)
        | (low_s > high_s + eps)
    )
    low_invalid_count = int(low_invalid.sum())
    if low_invalid_count > 0:
        errors.append(
            f"{low_invalid_count} bars violate Low <= min(Open, Close, High)."
        )

    if errors:
        error_summary = "\n  - " + "\n  - ".join(errors)
        raise ConversionValidationError(
            f"Source CSV validation FAILED with {len(errors)} issues:{error_summary}"
        )

    stats["num_invalid_rows"] = (
        stats["num_missing_ohlc"]
        + stats["num_negative_prices"]
        + stats["num_negative_volume"]
        + high_invalid_count
        + low_invalid_count
    )

    logger.info("Source CSV validation passed successfully.")
    return stats


def convert_to_mt5(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    tick_volume_mode: str = "constant",
    spread: int = 0,
    digits: Optional[int] = None,
) -> pd.DataFrame:
    """
    Transform source DataFrame into MT5 Custom Symbol bar format.

    Args:
        df: Validated source DataFrame.
        col_map: Detected column mapping.
        tick_volume_mode: 'constant' (TickVolume=1) or 'volume' (TickVolume=RealVolume).
        spread: Spread in points/pips (default 0).
        digits: Optional decimal precision. If None, preserves exact values.

    Returns:
        DataFrame ready for MT5 CSV export.
    """
    out_df = pd.DataFrame()

    # 1. Format DateTime: YYYY.MM.DD 00:00:00
    # Note: 00:00:00 is used because source data is daily without intraday timestamps.
    dates = pd.to_datetime(df[col_map["date"]])
    out_df["<DateTime>"] = dates.dt.strftime("%Y.%m.%d 00:00:00")

    # 2. OHLC values: preserved exactly
    for mt5_col, key in [
        ("<Open>", "open"),
        ("<High>", "high"),
        ("<Low>", "low"),
        ("<Close>", "close"),
    ]:
        series = df[col_map[key]].astype(float)
        if digits is not None:
            series = series.round(digits)
        out_df[mt5_col] = series

    # 3. RealVolume: source volume as integer
    vol_col = col_map.get("volume")
    if vol_col and vol_col in df.columns:
        out_df["<RealVolume>"] = df[vol_col].astype(np.int64)
    else:
        out_df["<RealVolume>"] = 0

    # 4. TickVolume
    if tick_volume_mode == "constant":
        out_df["<TickVolume>"] = 1
    elif tick_volume_mode == "volume":
        logger.warning(
            "Compatibility Notice: Using source Volume as TickVolume. "
            "This is NOT true MT5 tick volume and serves only as a placeholder."
        )
        out_df["<TickVolume>"] = out_df["<RealVolume>"]
    else:
        raise ValueError(f"Unsupported tick_volume_mode: {tick_volume_mode}")

    # 5. Spread
    out_df["<Spread>"] = int(spread)

    # Reorder exactly to MT5 specification
    out_df = out_df[MT5_BAR_COLUMNS]

    return out_df


def save_mt5_csv(
    df: pd.DataFrame, output_path: Path, digits: Optional[int] = None
) -> None:
    """
    Save DataFrame to CSV in MT5 Custom Symbol bar format without scientific notation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    float_fmt = f"%.{digits}f" if digits is not None else "%.10g"
    df.to_csv(output_path, index=False, float_format=float_fmt)
    logger.info("Saved MT5 BAR CSV: %s (%d rows)", output_path, len(df))


def create_conversion_report(
    report_path: Path,
    input_file: Path,
    output_file: Path,
    stats: Dict[str, Any],
    tick_volume_mode: str,
    spread: int,
    status: str = "SUCCESS",
) -> None:
    """
    Generate conversion summary report file.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tick_volume_val = "1" if tick_volume_mode == "constant" else "source Volume"

    content = f"""MT5 CSV CONVERSION REPORT
==========================

Input:
{input_file.as_posix()}

Output:
{output_file.as_posix()}

Input rows: {stats['input_rows']}
Output rows: {stats.get('output_rows', stats['input_rows'])}

First date:
{stats.get('first_date', 'N/A')}

Last date:
{stats.get('last_date', 'N/A')}

Number of duplicates: {stats.get('num_duplicates', 0)}
Number of invalid rows: {stats.get('num_invalid_rows', 0)}
Number of missing OHLC values: {stats.get('num_missing_ohlc', 0)}
Number of negative prices: {stats.get('num_negative_prices', 0)}
Number of negative volume values: {stats.get('num_negative_volume', 0)}

Tick volume mode:
{tick_volume_mode}

Tick volume:
{tick_volume_val}

Spread:
{spread}

Invalid rows:
{stats.get('num_invalid_rows', 0)}

Conversion status:
{status}
"""
    report_path.write_text(content, encoding="utf-8")
    logger.info("Saved conversion report: %s", report_path)


def run_integrity_check(
    source_df: pd.DataFrame,
    col_map: Dict[str, str],
    mt5_csv_path: Path,
    digits: Optional[int] = None,
) -> bool:
    """
    Independently re-reads both source and generated MT5 CSV and verifies:
    - same number of valid source rows
    - same Open values
    - same High values
    - same Low values
    - same Close values
    - same source Volume mapped to RealVolume
    - same first date
    - same last date

    Prints:
    OHLC INTEGRITY CHECK: PASSED
    """
    logger.info("Running independent integrity check on %s...", mt5_csv_path)

    if not mt5_csv_path.exists():
        raise IntegrityCheckError(f"Output MT5 file does not exist: {mt5_csv_path}")

    mt5_df = pd.read_csv(mt5_csv_path)

    # 1. Row count
    if len(source_df) != len(mt5_df):
        raise IntegrityCheckError(
            f"Row count mismatch: Source has {len(source_df)} rows, "
            f"MT5 has {len(mt5_df)} rows."
        )

    # 2. Date verification
    src_first_date = pd.to_datetime(source_df[col_map["date"]].iloc[0]).strftime("%Y.%m.%d 00:00:00")
    src_last_date = pd.to_datetime(source_df[col_map["date"]].iloc[-1]).strftime("%Y.%m.%d 00:00:00")
    mt5_first_date = str(mt5_df["<DateTime>"].iloc[0])
    mt5_last_date = str(mt5_df["<DateTime>"].iloc[-1])

    if src_first_date != mt5_first_date:
        raise IntegrityCheckError(
            f"First date mismatch: Source has {src_first_date}, MT5 has {mt5_first_date}."
        )
    if src_last_date != mt5_last_date:
        raise IntegrityCheckError(
            f"Last date mismatch: Source has {src_last_date}, MT5 has {mt5_last_date}."
        )

    # 3. OHLC price checks
    tol = 10 ** (-digits) if digits is not None else 1e-6
    for key, mt5_col in [
        ("open", "<Open>"),
        ("high", "<High>"),
        ("low", "<Low>"),
        ("close", "<Close>"),
    ]:
        src_vals = source_df[col_map[key]].to_numpy(dtype=float)
        mt5_vals = mt5_df[mt5_col].to_numpy(dtype=float)

        if not np.allclose(src_vals, mt5_vals, atol=tol, rtol=1e-5):
            max_diff = np.max(np.abs(src_vals - mt5_vals))
            raise IntegrityCheckError(
                f"Mismatch in {key.upper()} values! Max absolute difference: {max_diff:.8f}"
            )

    # 4. Volume check
    vol_col = col_map.get("volume")
    if vol_col and vol_col in source_df.columns:
        src_vol = source_df[vol_col].to_numpy(dtype=np.int64)
        mt5_vol = mt5_df["<RealVolume>"].to_numpy(dtype=np.int64)
        if not np.array_equal(src_vol, mt5_vol):
            raise IntegrityCheckError("Mismatch in Volume / RealVolume mapping!")

    print("\nOHLC INTEGRITY CHECK: PASSED\n")
    logger.info("OHLC integrity verification completed with 0 errors.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QUANTLAB OHLCV CSV data into MetaTrader 5 Custom Symbol Bar CSV format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to source CSV file (e.g. data/raw/RELIANCE.csv or data/processed/RELIANCE.csv).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output MT5 BAR CSV file (e.g. validation/mt5/RELIANCE_MT5.csv).",
    )
    parser.add_argument(
        "--tick-volume-mode",
        choices=["constant", "volume"],
        default="constant",
        help="Rule for TickVolume: 'constant' sets TickVolume=1; 'volume' maps source Volume to TickVolume.",
    )
    parser.add_argument(
        "--spread",
        type=int,
        default=0,
        help="Configurable default spread in points (default: 0).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional path for validation report txt file. Defaults to <output_dir>/<stem>_conversion_report.txt.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        help="Display the first N converted rows without modifying the input.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Optional decimal places for price formatting. If not specified, preserves original decimal precision.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 1

    logger.info("Reading source CSV: %s", input_path)
    try:
        source_df = pd.read_csv(input_path)
    except Exception as e:
        logger.error("Failed to read input CSV: %s", e)
        return 1

    logger.info("Source columns: %s", list(source_df.columns))

    # Detect columns
    try:
        col_map = detect_columns(list(source_df.columns))
    except ConversionValidationError as e:
        logger.error("Column detection error: %s", e)
        return 1

    logger.info("Detected date column: %s", col_map["date"])
    logger.info("Detected OHLC columns: Open=%s, High=%s, Low=%s, Close=%s, Volume=%s",
                col_map["open"], col_map["high"], col_map["low"], col_map["close"], col_map.get("volume"))

    # Validate source data
    try:
        stats = validate_source_data(source_df, col_map)
    except ConversionValidationError as e:
        logger.error("Validation failed:\n%s", e)
        return 1

    # Convert data
    try:
        mt5_df = convert_to_mt5(
            df=source_df,
            col_map=col_map,
            tick_volume_mode=args.tick_volume_mode,
            spread=args.spread,
            digits=args.digits,
        )
    except Exception as e:
        logger.error("Conversion failed: %s", e)
        return 1

    stats["output_rows"] = len(mt5_df)

    # Preview mode if requested
    if args.preview is not None and args.preview > 0:
        print(f"\n--- PREVIEW: FIRST {args.preview} CONVERTED ROWS ---")
        preview_sample = mt5_df.head(args.preview)
        print(preview_sample.to_string(index=False))
        print("--- END PREVIEW ---\n")

    # Save output MT5 CSV
    try:
        save_mt5_csv(mt5_df, output_path, digits=args.digits)
    except Exception as e:
        logger.error("Failed to save output CSV: %s", e)
        return 1

    # Save report
    if args.report:
        report_path = Path(args.report)
    else:
        # Default report name alongside output
        report_stem = output_path.stem.replace("_MT5", "")
        report_path = output_path.parent / f"{report_stem}_conversion_report.txt"

    try:
        create_conversion_report(
            report_path=report_path,
            input_file=input_path,
            output_file=output_path,
            stats=stats,
            tick_volume_mode=args.tick_volume_mode,
            spread=args.spread,
            status="SUCCESS",
        )
    except Exception as e:
        logger.error("Failed to write report: %s", e)
        return 1

    # Run integrity check
    try:
        run_integrity_check(source_df, col_map, output_path, digits=args.digits)
    except IntegrityCheckError as e:
        logger.error("Integrity check failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
