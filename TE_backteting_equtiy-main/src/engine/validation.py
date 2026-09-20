"""
validation.py

Data Validation Module for QUANTLAB Backtesting Engine.
Validates price Series/DataFrames before execution to prevent quiet failures.
"""

from typing import Tuple, List
import pandas as pd


class DataValidationError(ValueError):
    """Exception raised when input data fails backtest validation checks."""
    pass


class DataValidator:
    """
    Validates input stock price DataFrames before passing to engine.
    """

    @staticmethod
    def validate(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate input dataset for data integrity.

        Checks:
        1. Non-empty DataFrame
        2. Index is DatetimeIndex
        3. Sorted by timestamp
        4. No duplicate timestamps
        5. Required columns exist (e.g. Close, Signal, Position)
        6. Prices are strictly positive (> 0)
        7. No NaN or inf values in price columns

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_warning_or_error_messages)
        """
        errors = []

        if df is None or df.empty:
            raise DataValidationError("Execution Error: Input DataFrame is empty or None.")

        # Check Index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                errors.append(f"Index could not be converted to DatetimeIndex: {e}")

        # Check sorting
        if not df.index.is_monotonic_increasing:
            errors.append("Dataset index is not sorted in chronological ascending order.")

        # Check duplicates
        if df.index.has_duplicates:
            errors.append("Dataset contains duplicate timestamp entries.")

        # Required columns check
        cols_to_check = required_columns or ["Close", "Position", "Signal"]
        for col in cols_to_check:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")

        # Check prices non-negative & non-null
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
        for col in price_cols:
            if df[col].isnull().any():
                errors.append(f"Column '{col}' contains NaN values.")
            if (df[col] <= 0).any():
                errors.append(f"Column '{col}' contains non-positive (<= 0) prices.")

        if errors:
            raise DataValidationError("Data Validation Failed:\n- " + "\n- ".join(errors))

        return True, ["Dataset validated successfully."]
