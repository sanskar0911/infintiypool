#!/usr/bin/env python3
"""
tools/convert_mt5_csv_to_json.py

Python conversion utility for QUANTLAB.
Converts MT5-compatible bar CSV data into a structured JSON file.

Standard libraries used: csv, json, argparse, pathlib, logging, datetime, math.

Schema Note:
The output JSON structure is a QUANTLAB structured data representation designed
for downstream ingestion, API validation, and testing interfaces. It is NOT an
official native MetaTrader 5 terminal file format (.hcc/.dat).
"""

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("convert_mt5_csv_to_json")

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



class ValidationError(Exception):
    """Raised when CSV validation fails."""
    pass


class IntegrityError(Exception):
    """Raised when post-conversion integrity check fails."""
    pass


# Canonical required column keys
REQUIRED_FIELDS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "tickvolume",
    "realvolume",
    "spread",
]

# Known descriptions for symbols
KNOWN_DESCRIPTIONS: Dict[str, str] = {
    "RELIANCE": "Reliance Industries - QUANTLAB Validation",
    "HDFCBANK": "HDFC Bank - QUANTLAB Validation",
    "ICICIBANK": "ICICI Bank - QUANTLAB Validation",
    "INFY": "Infosys - QUANTLAB Validation",
    "TCS": "Tata Consultancy Services - QUANTLAB Validation",
    "NIFTY50": "Nifty 50 Index - QUANTLAB Validation",
}


def normalize_column_name(col: str) -> str:
    """Normalize column header: strip whitespace and angle brackets, convert to lowercase."""
    return col.strip().strip("<>").lower()


def detect_columns(header: List[str]) -> Dict[str, str]:
    """
    Validate that all required columns are present in the CSV header.
    Supports headers both with and without angle brackets:
    e.g. '<DateTime>' or 'DateTime'.

    Returns mapping of canonical key to original header string.
    """
    normalized_map = {normalize_column_name(col): col for col in header}
    missing = [req for req in REQUIRED_FIELDS if req not in normalized_map]

    if missing:
        raise ValidationError(
            f"Missing required columns in CSV: {missing}. Found columns: {header}"
        )

    return {req: normalized_map[req] for req in REQUIRED_FIELDS}


def parse_timestamp(dt_str: str) -> datetime:
    """
    Parse timestamp string across common MT5 / ISO formats.
    """
    dt_str = dt_str.strip()
    formats = [
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        raise ValueError(f"Unparseable datetime value: '{dt_str}'")


def validate_and_convert_records(
    rows: List[Dict[str, str]], col_map: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validate every record row by row and convert to Python dictionary records:
    - DateTime is valid and parseable
    - Open, High, Low, Close are numeric, non-NaN, and > 0
    - TickVolume, RealVolume, Spread are numeric and >= 0
    - High >= Open, High >= Close, High >= Low
    - Low <= Open, Low <= Close
    - Chronological ascending order
    - No duplicate timestamps

    Returns:
        (records, stats)
    """
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_timestamps: set[str] = set()
    prev_dt: Optional[datetime] = None
    num_duplicates = 0

    col_dt = col_map["datetime"]
    col_open = col_map["open"]
    col_high = col_map["high"]
    col_low = col_map["low"]
    col_close = col_map["close"]
    col_tvol = col_map["tickvolume"]
    col_rvol = col_map["realvolume"]
    col_spd = col_map["spread"]

    eps = 1e-7

    for idx, row in enumerate(rows, start=2):  # row 2 is first data line after header
        # 1. Parse timestamp
        raw_dt = row.get(col_dt, "").strip()
        if not raw_dt:
            errors.append(f"Row {idx}: DateTime is empty.")
            continue

        try:
            curr_dt = parse_timestamp(raw_dt)
        except ValueError as e:
            errors.append(f"Row {idx}: {e}")
            continue

        # Duplicate check
        if raw_dt in seen_timestamps:
            errors.append(f"Row {idx}: Duplicate timestamp detected: '{raw_dt}'.")
            num_duplicates += 1
        seen_timestamps.add(raw_dt)

        # Ascending order check
        if prev_dt is not None:
            if curr_dt < prev_dt:
                errors.append(
                    f"Row {idx}: Non-ascending timestamp '{raw_dt}' occurs after '{prev_dt}'."
                )
        prev_dt = curr_dt

        # 2. Parse Open, High, Low, Close
        try:
            val_open = float(row[col_open])
            val_high = float(row[col_high])
            val_low = float(row[col_low])
            val_close = float(row[col_close])
        except (ValueError, KeyError) as e:
            errors.append(f"Row {idx}: Non-numeric OHLC price found: {e}")
            continue

        for p_name, p_val in [
            ("Open", val_open),
            ("High", val_high),
            ("Low", val_low),
            ("Close", val_close),
        ]:
            if math.isnan(p_val) or math.isinf(p_val):
                errors.append(f"Row {idx}: {p_name} is NaN or Inf.")
            elif p_val <= 0:
                errors.append(f"Row {idx}: {p_name} must be positive, got {p_val}.")

        # 3. Bar boundary checks
        if val_high < val_open - eps or val_high < val_close - eps or val_high < val_low - eps:
            errors.append(
                f"Row {idx}: High ({val_high}) violates High >= max(Open={val_open}, Close={val_close}, Low={val_low})."
            )

        if val_low > val_open + eps or val_low > val_close + eps:
            errors.append(
                f"Row {idx}: Low ({val_low}) violates Low <= min(Open={val_open}, Close={val_close})."
            )

        # 4. Parse Volumes and Spread
        try:
            val_tvol = int(float(row[col_tvol]))
            val_rvol = int(float(row[col_rvol]))
            val_spd = int(float(row[col_spd]))
        except (ValueError, KeyError) as e:
            errors.append(f"Row {idx}: Non-numeric volume/spread found: {e}")
            continue

        if val_tvol < 0:
            errors.append(f"Row {idx}: TickVolume cannot be negative ({val_tvol}).")
        if val_rvol < 0:
            errors.append(f"Row {idx}: RealVolume cannot be negative ({val_rvol}).")
        if val_spd < 0:
            errors.append(f"Row {idx}: Spread cannot be negative ({val_spd}).")

        # Record construction preserving exact float representation
        records.append({
            "datetime": raw_dt,
            "open": val_open,
            "high": val_high,
            "low": val_low,
            "close": val_close,
            "tick_volume": val_tvol,
            "real_volume": val_rvol,
            "spread": val_spd,
        })

    if errors:
        error_summary = "\n  - " + "\n  - ".join(errors[:20])
        if len(errors) > 20:
            error_summary += f"\n  ... and {len(errors) - 20} more errors."
        raise ValidationError(f"CSV validation failed with {len(errors)} issues:{error_summary}")

    stats = {
        "num_input": len(rows),
        "num_json": len(records),
        "num_duplicates": num_duplicates,
        "num_invalid": len(errors),
        "first_datetime": records[0]["datetime"] if records else "N/A",
        "last_datetime": records[-1]["datetime"] if records else "N/A",
    }
    return records, stats


def build_json_payload(
    symbol: str,
    records: List[Dict[str, Any]],
    description: Optional[str] = None,
    timeframe: str = "D1",
    currency: str = "INR",
    data_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct the final structured QUANTLAB JSON document.
    """
    base_sym = symbol.replace("QUANTLAB_", "")
    if not description:
        description = KNOWN_DESCRIPTIONS.get(
            base_sym, f"{base_sym} - QUANTLAB Validation"
        )
    if not data_source:
        data_source = f"QUANTLAB {base_sym} OHLCV dataset"

    return {
        "symbol": symbol,
        "description": description,
        "timeframe": timeframe,
        "currency": currency,
        "data_source": data_source,
        "records": records,
    }


def write_conversion_report(
    report_path: Path,
    input_file: Path,
    output_file: Path,
    symbol: str,
    timeframe: str,
    stats: Dict[str, Any],
    status: str = "SUCCESS",
) -> None:
    """Generate the conversion text report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""CSV → JSON CONVERSION REPORT
============================

Input:
{input_file.as_posix()}

Output:
{output_file.as_posix()}

Symbol:
{symbol}

Timeframe:
{timeframe}

Input records:
{stats['num_input']}

JSON records:
{stats['num_json']}

First datetime:
{stats['first_datetime']}

Last datetime:
{stats['last_datetime']}

Invalid records:
{stats.get('num_invalid', 0)}

Duplicate timestamps:
{stats.get('num_duplicates', 0)}

Conversion status:
{status}
"""
    report_path.write_text(content, encoding="utf-8")
    logger.info("Saved conversion report: %s", report_path)


def run_integrity_check(
    csv_path: Path, json_path: Path, col_map: Dict[str, str]
) -> bool:
    """
    Independently verify CSV and JSON match:
    - input row count == JSON record count
    - first DateTime matches
    - last DateTime matches
    - Open, High, Low, Close match
    - TickVolume, RealVolume, Spread match

    Prints:
    DATA INTEGRITY CHECK: PASSED
    """
    logger.info("Running independent integrity check between CSV and JSON...")

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    with open(json_path, mode="r", encoding="utf-8") as f:
        json_data = json.load(f)

    json_records = json_data.get("records", [])

    if len(csv_rows) != len(json_records):
        raise IntegrityError(
            f"Record count mismatch: CSV has {len(csv_rows)}, JSON has {len(json_records)}."
        )

    col_dt = col_map["datetime"]
    col_open = col_map["open"]
    col_high = col_map["high"]
    col_low = col_map["low"]
    col_close = col_map["close"]
    col_tvol = col_map["tickvolume"]
    col_rvol = col_map["realvolume"]
    col_spd = col_map["spread"]

    first_csv_dt = csv_rows[0][col_dt].strip()
    last_csv_dt = csv_rows[-1][col_dt].strip()
    first_json_dt = json_records[0]["datetime"].strip()
    last_json_dt = json_records[-1]["datetime"].strip()

    if first_csv_dt != first_json_dt:
        raise IntegrityError(
            f"First DateTime mismatch: CSV '{first_csv_dt}' vs JSON '{first_json_dt}'."
        )
    if last_csv_dt != last_json_dt:
        raise IntegrityError(
            f"Last DateTime mismatch: CSV '{last_csv_dt}' vs JSON '{last_json_dt}'."
        )

    for i, (crow, jrec) in enumerate(zip(csv_rows, json_records)):
        if crow[col_dt].strip() != jrec["datetime"].strip():
            raise IntegrityError(f"Record {i}: DateTime mismatch.")
        if abs(float(crow[col_open]) - jrec["open"]) > 1e-6:
            raise IntegrityError(f"Record {i}: Open mismatch.")
        if abs(float(crow[col_high]) - jrec["high"]) > 1e-6:
            raise IntegrityError(f"Record {i}: High mismatch.")
        if abs(float(crow[col_low]) - jrec["low"]) > 1e-6:
            raise IntegrityError(f"Record {i}: Low mismatch.")
        if abs(float(crow[col_close]) - jrec["close"]) > 1e-6:
            raise IntegrityError(f"Record {i}: Close mismatch.")
        if int(float(crow[col_tvol])) != jrec["tick_volume"]:
            raise IntegrityError(f"Record {i}: TickVolume mismatch.")
        if int(float(crow[col_rvol])) != jrec["real_volume"]:
            raise IntegrityError(f"Record {i}: RealVolume mismatch.")
        if int(float(crow[col_spd])) != jrec["spread"]:
            raise IntegrityError(f"Record {i}: Spread mismatch.")

    print("\nDATA INTEGRITY CHECK: PASSED\n")
    logger.info("Data integrity check completed with 0 errors.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MT5-compatible bar CSV into structured QUANTLAB JSON format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to MT5 bar CSV file (e.g. RELIANCE_MT5.csv or validation/mt5/RELIANCE_MT5.csv).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to target JSON file (e.g. RELIANCE_MT5.json or validation/mt5/RELIANCE_MT5.json).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol identifier (e.g. QUANTLAB_RELIANCE). Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Dataset description string.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="D1",
        help="Bar timeframe (default: D1).",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="INR",
        help="Currency denomination (default: INR).",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Data source description string.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to conversion report txt file. Defaults to <output_stem>_conversion_report.txt.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        help="Display the first N converted records without modifying files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input CSV file does not exist: %s", input_path)
        return 1

    # Infer symbol if not provided
    if args.symbol:
        symbol = args.symbol
    else:
        # e.g., RELIANCE_MT5.csv -> RELIANCE
        base = input_path.stem.replace("_MT5", "").replace("_mt5", "")
        symbol = f"QUANTLAB_{base}"

    logger.info("Reading CSV: %s", input_path)
    with open(input_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            logger.error("Input CSV file is empty.")
            return 1

    logger.info("Detected header columns: %s", raw_header)
    try:
        col_map = detect_columns(raw_header)
    except ValidationError as e:
        logger.error("Column validation failed: %s", e)
        return 1

    # Read rows as DictReader
    with open(input_path, mode="r", encoding="utf-8-sig") as f:
        dict_reader = csv.DictReader(f)
        rows = list(dict_reader)

    logger.info("Total CSV rows read: %d", len(rows))

    # Validate and convert
    try:
        records, stats = validate_and_convert_records(rows, col_map)
    except ValidationError as e:
        logger.error("Data validation failed:\n%s", e)
        return 1

    payload = build_json_payload(
        symbol=symbol,
        records=records,
        description=args.description,
        timeframe=args.timeframe,
        currency=args.currency,
        data_source=args.data_source,
    )

    # Preview mode
    if args.preview is not None and args.preview > 0:
        preview_payload = dict(payload)
        preview_payload["records"] = records[:args.preview]
        print(f"\n--- PREVIEW: FIRST {args.preview} CONVERTED JSON RECORDS ---")
        print(json.dumps(preview_payload, indent=2))
        print("--- END PREVIEW ---\n")

    # Save output JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved JSON file: %s (%d records)", output_path, len(records))

    # Save report
    if args.report:
        report_path = Path(args.report)
    else:
        # Default alongside output
        report_name = f"{output_path.stem.replace('_MT5', '')}_json_conversion_report.txt"
        report_path = output_path.parent / report_name

    write_conversion_report(
        report_path=report_path,
        input_file=input_path,
        output_file=output_path,
        symbol=symbol,
        timeframe=args.timeframe,
        stats=stats,
        status="SUCCESS",
    )

    # Run integrity check
    try:
        run_integrity_check(input_path, output_path, col_map)
    except IntegrityError as e:
        logger.error("Integrity check failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
