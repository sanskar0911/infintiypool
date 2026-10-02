"""Input Excel Processing and Data Validation for QuantLab Backtesting Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import openpyxl
import pandas as pd
import numpy as np


COLUMN_ALIASES: Dict[str, List[str]] = {
    "stock": [
        "stock", "symbol", "ticker", "instrument", "name", "stock name",
        "company", "security"
    ],
    "entry_date": [
        "entry date", "entrydate", "buy date", "start date", "entry_date",
        "startdate", "trade date", "buy_date"
    ],
    "entry_price": [
        "entry price", "entryprice", "buy price", "entry_price", "buyprice",
        "purchase price"
    ],
    "exit_date": [
        "exit date", "exitdate", "sell date", "end date", "exit_date",
        "enddate", "sell_date"
    ],
    "exit_price": [
        "close price", "closeprice", "exit price", "exitprice", "sell price",
        "closing price", "close_price", "exit_price", "sell_price"
    ],
    "provided_return": [
        "returns delivered (%)", "returns delivered", "return", "returns",
        "p&l %", "return %", "returns %", "pnl %", "profit %", "ret %"
    ],
    "direction": [
        "direction", "type", "side", "trade type", "position"
    ]
}


@dataclass
class ValidationReport:
    """Stores input validation summary statistics."""
    file_path: str
    sheet_name: str
    total_raw_rows: int
    data_rows: int
    columns: List[str]
    unique_stocks: int
    start_date: Optional[str]
    end_date: Optional[str]
    missing_values: int
    duplicate_trades: int
    invalid_prices: int
    invalid_dates: int
    invalid_returns: int
    active_trades: int
    issues: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 40)
        print("INPUT VALIDATION")
        print("-" * 40)
        print(f"Input File:       {Path(self.file_path).name}")
        print(f"Sheet:            {self.sheet_name}")
        print(f"Rows:             {self.data_rows}")
        print(f"Columns:          {', '.join(self.columns)}")
        print(f"Stocks:           {self.unique_stocks}")
        print(f"Start:            {self.start_date or 'N/A'}")
        print(f"End:              {self.end_date or 'N/A'}")
        print(f"Active trades:    {self.active_trades}")
        print(f"Missing values:   {self.missing_values}")
        print(f"Duplicate trades: {self.duplicate_trades}")
        print(f"Invalid prices:   {self.invalid_prices}")
        print(f"Invalid dates:    {self.invalid_dates}")
        print(f"Invalid returns:  {self.invalid_returns}")
        print("-" * 40)
        if self.issues:
            print("Validation Notices/Issues:")
            for issue in self.issues[:10]:
                print(f" - {issue}")
            if len(self.issues) > 10:
                print(f" ... and {len(self.issues) - 10} more issues.")
        print("=" * 40 + "\n")


def clean_cell_text(val: Any) -> str:
    """Convert cell value to lowercase normalized string for header matching."""
    if val is None:
        return ""
    return str(val).strip().lower()


def find_header_row(rows: List[List[Any]]) -> Tuple[int, Dict[int, str]]:
    """Automatically find header row by matching keywords against COLUMN_ALIASES."""
    best_row_idx = -1
    best_mapping: Dict[int, str] = {}
    max_matches = 0

    for idx, row in enumerate(rows[:25]):
        current_mapping: Dict[int, str] = {}
        for col_idx, cell in enumerate(row):
            txt = clean_cell_text(cell)
            if not txt:
                continue
            for standard_col, aliases in COLUMN_ALIASES.items():
                if standard_col in current_mapping.values():
                    continue
                if txt in aliases or any(txt == a.lower() for a in aliases):
                    current_mapping[col_idx] = standard_col
                    break

        # We need at least stock, entry_date, entry_price to consider it a valid header
        matches = len(current_mapping)
        if matches >= 3 and matches > max_matches:
            max_matches = matches
            best_row_idx = idx
            best_mapping = current_mapping

    if best_row_idx == -1:
        raise ValueError("Could not automatically detect a valid header row in the Excel sheet.")

    return best_row_idx, best_mapping


def parse_trade_date(val: Any) -> Tuple[Optional[pd.Timestamp], str, bool]:
    """Parse date from value, handling strings like 'Active(on Jul 31, 2024)'."""
    if val is None or pd.isna(val):
        return None, "", False

    raw_str = str(val).strip()
    is_active = False

    cleaned_str = raw_str
    if "active" in raw_str.lower():
        is_active = True
        # Match pattern Active(on <date>) or Active (<date>)
        m = re.search(r"active\s*\(?(?:on\s*)?([^\)]+)\)?", raw_str, flags=re.IGNORECASE)
        if m:
            cleaned_str = m.group(1).strip()
        else:
            cleaned_str = re.sub(r"active", "", raw_str, flags=re.IGNORECASE).strip("() ")

    try:
        ts = pd.to_datetime(cleaned_str, errors="coerce")
        if pd.isna(ts):
            return None, raw_str, is_active
        return ts, raw_str, is_active
    except Exception:
        return None, raw_str, is_active


def parse_numeric(val: Any) -> Optional[float]:
    """Parse numeric price safely, stripping currency signs and commas."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace(",", "").replace("₹", "").replace("$", "").replace("INR", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_percentage(val: Any) -> Optional[float]:
    """Parse percentage value safely, e.g. '5.25%', '5.25', '-2.31%', NaN."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def load_signal_file(
    file_path: str,
    sheet_name: Optional[str] = None
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Read .xlsx signal file, detect sheets & headers, validate data, and return cleaned DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    wb = openpyxl.load_workbook(file_path, data_only=True)
    available_sheets = wb.sheetnames

    if sheet_name is None:
        sheet_name = available_sheets[0]
    elif sheet_name not in available_sheets:
        raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}")

    sheet = wb[sheet_name]
    raw_rows = list(sheet.iter_rows(values_only=True))

    if not raw_rows:
        raise ValueError(f"Sheet '{sheet_name}' is empty.")

    header_idx, col_mapping = find_header_row(raw_rows)
    required_cols = {"stock", "entry_date", "entry_price", "exit_date", "exit_price"}
    missing_required = required_cols - set(col_mapping.values())
    if missing_required:
        raise ValueError(f"Missing required columns in signal file: {missing_required}")

    # Extract data rows after header
    data_records: List[Dict[str, Any]] = []
    issues: List[str] = []
    invalid_prices_cnt = 0
    invalid_dates_cnt = 0
    invalid_returns_cnt = 0
    missing_values_cnt = 0
    active_trades_cnt = 0

    for r_idx in range(header_idx + 1, len(raw_rows)):
        row = raw_rows[r_idx]
        if not any(row):  # skip completely blank rows
            continue

        record: Dict[str, Any] = {"_raw_row_idx": r_idx + 1}
        for col_idx, col_name in col_mapping.items():
            cell_val = row[col_idx] if col_idx < len(row) else None
            record[col_name] = cell_val

        stock_val = record.get("stock")
        if not stock_val or pd.isna(stock_val):
            missing_values_cnt += 1
            issues.append(f"Row {r_idx + 1}: Missing stock name.")
            continue
        stock_clean = str(stock_val).strip()

        # Parse Entry Date
        entry_ts, entry_raw, _ = parse_trade_date(record.get("entry_date"))
        if entry_ts is None:
            invalid_dates_cnt += 1
            issues.append(f"Row {r_idx + 1} ({stock_clean}): Invalid entry date '{record.get('entry_date')}'.")

        # Parse Exit Date
        exit_ts, exit_raw, is_active = parse_trade_date(record.get("exit_date"))
        if exit_ts is None:
            invalid_dates_cnt += 1
            issues.append(f"Row {r_idx + 1} ({stock_clean}): Invalid exit date '{record.get('exit_date')}'.")
        if is_active:
            active_trades_cnt += 1

        # Parse Prices
        entry_price = parse_numeric(record.get("entry_price"))
        if entry_price is None or entry_price <= 0:
            invalid_prices_cnt += 1
            issues.append(f"Row {r_idx + 1} ({stock_clean}): Invalid entry price '{record.get('entry_price')}'.")

        exit_price = parse_numeric(record.get("exit_price"))
        if exit_price is None or exit_price <= 0:
            invalid_prices_cnt += 1
            issues.append(f"Row {r_idx + 1} ({stock_clean}): Invalid exit/close price '{record.get('exit_price')}'.")

        # Parse Provided Return
        prov_ret = parse_percentage(record.get("provided_return"))
        if record.get("provided_return") is not None and not pd.isna(record.get("provided_return")) and prov_ret is None:
            invalid_returns_cnt += 1
            issues.append(f"Row {r_idx + 1} ({stock_clean}): Non-numeric return '{record.get('provided_return')}'.")

        # Parse Direction
        dir_val = str(record.get("direction", "")).strip().upper() if record.get("direction") else "LONG"
        if dir_val not in ["LONG", "SHORT", "BUY", "SELL"]:
            dir_val = "LONG"
        if dir_val == "BUY":
            dir_val = "LONG"
        elif dir_val == "SELL":
            dir_val = "SHORT"

        cleaned_row = {
            "stock": stock_clean,
            "entry_date_raw": entry_raw,
            "entry_date": entry_ts,
            "entry_price": entry_price,
            "exit_date_raw": exit_raw,
            "exit_date": exit_ts,
            "exit_price": exit_price,
            "provided_return": prov_ret,
            "direction": dir_val,
            "is_active": is_active,
            "_raw_row_idx": r_idx + 1
        }
        data_records.append(cleaned_row)

    df = pd.DataFrame(data_records)

    # Check for critical errors
    if df.empty:
        raise ValueError(f"No valid trade rows found in '{file_path}'.")

    # Check duplicates
    duplicate_trades_cnt = int(df.duplicated(subset=["stock", "entry_date", "entry_price", "exit_date", "exit_price"]).sum())
    if duplicate_trades_cnt > 0:
        issues.append(f"Found {duplicate_trades_cnt} duplicate trade rows.")

    # Check missing required values
    missing_crit = df[["stock", "entry_date", "entry_price", "exit_date", "exit_price"]].isna().any(axis=1).sum()
    if missing_crit > 0:
        raise ValueError(f"Stopping execution: {missing_crit} rows have critical missing fields (date, price, or stock).")

    unique_stocks = df["stock"].nunique()
    start_date_str = df["entry_date"].min().strftime("%Y-%m-%d") if not df["entry_date"].isna().all() else None
    end_date_str = df["exit_date"].max().strftime("%Y-%m-%d") if not df["exit_date"].isna().all() else None

    report = ValidationReport(
        file_path=str(path),
        sheet_name=sheet_name,
        total_raw_rows=len(raw_rows),
        data_rows=len(df),
        columns=list(col_mapping.values()),
        unique_stocks=unique_stocks,
        start_date=start_date_str,
        end_date=end_date_str,
        missing_values=missing_values_cnt,
        duplicate_trades=duplicate_trades_cnt,
        invalid_prices=invalid_prices_cnt,
        invalid_dates=invalid_dates_cnt,
        invalid_returns=invalid_returns_cnt,
        active_trades=active_trades_cnt,
        issues=issues
    )

    return df, report
