"""
trade_log.py

Trade Logging module for QUANTLAB Backtesting Engine.
Records canonical trade execution entries and formats output DataFrame.
"""

from typing import List, Dict, Any
import pandas as pd
from .fill import Fill


class TradeLogger:
    """
    Maintains chronological execution log for all orders filled during backtest.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log_trade(
        self,
        fill: Fill,
        cash_before: float,
        cash_after: float,
        portfolio_value: float,
        reason: str = "Signal"
    ):
        """Record an executed trade fill."""
        record = {
            "Date": fill.timestamp,
            "Symbol": fill.symbol,
            "BUY/SELL": str(fill.side),
            "Type": str(fill.side),
            "Shares": fill.quantity,
            "Price": fill.reference_price,
            "Execution Price": fill.fill_price,
            "Slippage": fill.slippage_cost,
            "Commission": fill.transaction_cost,
            "Transaction Cost": fill.transaction_cost,
            "Value": fill.gross_value,
            "Gross Value": fill.gross_value,
            "Net Value": fill.net_value,
            "Cash Before": cash_before,
            "Cash After": cash_after,
            "Cash": cash_after,
            "Remaining Cash": cash_after,
            "Portfolio Value": portfolio_value,
            "Reason": reason,
            "Order ID": fill.order_id,
            "Fill ID": fill.fill_id
        }
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade log records to pandas DataFrame indexed by Date."""
        if not self.records:
            cols = [
                "BUY/SELL", "Type", "Symbol", "Shares", "Price", "Execution Price",
                "Slippage", "Commission", "Transaction Cost", "Value", "Gross Value",
                "Net Value", "Cash Before", "Cash After", "Cash", "Remaining Cash",
                "Portfolio Value", "Reason", "Order ID", "Fill ID"
            ]
            df = pd.DataFrame(columns=cols)
            df.index.name = "Date"
            return df

        df = pd.DataFrame(self.records)
        df.set_index("Date", inplace=True)
        return df
