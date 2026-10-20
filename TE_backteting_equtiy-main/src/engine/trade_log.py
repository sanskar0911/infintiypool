"""
trade_log.py

Auditable Portfolio Ledger and Trade Logging for QUANTLAB Engine v1.1.
Maintains full forensic trail: timestamps, symbols, order/fill IDs, side, quantity,
prices, gross value, brokerage, STT, stamp duty, GST, exchange charges, slippage,
net cash movement, position tracking, cash balance, realized P&L, cumulative P&L,
portfolio equity, order status, and execution reasons.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from .fill import Fill
from .order import OrderSide


class TradeLogger:
    """
    Maintains the auditable ledger for all executions and order updates.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.cumulative_realized_pnl: float = 0.0

    def log_trade(
        self,
        fill: Fill,
        cash_before: float,
        cash_after: float,
        portfolio_value: float,
        position_before: int = 0,
        position_after: int = 0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        order_status: str = "FILLED",
        rejection_reason: str = "",
        reason: str = "Signal"
    ):
        """Record an executed trade fill or status transition in the auditable ledger."""
        if realized_pnl != 0.0:
            self.cumulative_realized_pnl += realized_pnl

        is_buy = (fill.side == OrderSide.BUY or str(fill.side).upper() == "BUY")
        net_cash_movement = -fill.net_value if is_buy else fill.net_value

        record = {
            "Date": fill.timestamp,
            "Symbol": fill.symbol,
            "BUY/SELL": str(fill.side),
            "Type": str(fill.side),
            "Side": str(fill.side),
            "Shares": fill.quantity,
            "Quantity": fill.quantity,
            "Price": fill.reference_price,
            "Reference Price": fill.reference_price,
            "Execution Price": fill.fill_price,
            "Executed Price": fill.fill_price,
            "Gross Value": fill.gross_value,
            "Value": fill.gross_value,
            "Slippage": fill.slippage_cost,
            "Brokerage": fill.brokerage,
            "STT": fill.stt,
            "Stamp Duty": fill.stamp_duty,
            "GST": fill.gst,
            "Exchange Charges": fill.exchange_charges,
            "Commission": fill.brokerage,
            "Transaction Cost": fill.transaction_cost,
            "Net Cash Movement": net_cash_movement,
            "Net Value": fill.net_value,
            "Position Before": position_before,
            "Position After": position_after,
            "Cash Before": cash_before,
            "Cash After": cash_after,
            "Cash": cash_after,
            "Remaining Cash": cash_after,
            "Realized PnL": realized_pnl,
            "Cumulative Realized PnL": self.cumulative_realized_pnl,
            "Unrealized PnL": unrealized_pnl,
            "Portfolio Value": portfolio_value,
            "Portfolio Equity": portfolio_value,
            "Order Status": order_status,
            "Rejection Reason": rejection_reason,
            "Reason": reason,
            "Order ID": fill.order_id,
            "Fill ID": fill.fill_id
        }
        self.records.append(record)

    def log_rejection(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        cash_available: float,
        portfolio_value: float,
        rejection_reason: str,
        order_id: str = ""
    ):
        """Record an order rejection in the auditable log."""
        record = {
            "Date": timestamp,
            "Symbol": symbol,
            "BUY/SELL": str(side),
            "Type": str(side),
            "Side": str(side),
            "Shares": quantity,
            "Quantity": quantity,
            "Price": price,
            "Reference Price": price,
            "Execution Price": 0.0,
            "Executed Price": 0.0,
            "Gross Value": 0.0,
            "Value": 0.0,
            "Slippage": 0.0,
            "Brokerage": 0.0,
            "STT": 0.0,
            "Stamp Duty": 0.0,
            "GST": 0.0,
            "Exchange Charges": 0.0,
            "Commission": 0.0,
            "Transaction Cost": 0.0,
            "Net Cash Movement": 0.0,
            "Net Value": 0.0,
            "Position Before": 0,
            "Position After": 0,
            "Cash Before": cash_available,
            "Cash After": cash_available,
            "Cash": cash_available,
            "Remaining Cash": cash_available,
            "Realized PnL": 0.0,
            "Cumulative Realized PnL": self.cumulative_realized_pnl,
            "Unrealized PnL": 0.0,
            "Portfolio Value": portfolio_value,
            "Portfolio Equity": portfolio_value,
            "Order Status": "REJECTED",
            "Rejection Reason": rejection_reason,
            "Reason": f"Rejected: {rejection_reason}",
            "Order ID": order_id,
            "Fill ID": "N/A"
        }
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade log records to pandas DataFrame indexed by Date."""
        if not self.records:
            cols = [
                "Symbol", "BUY/SELL", "Type", "Side", "Shares", "Quantity", "Price",
                "Reference Price", "Execution Price", "Executed Price", "Gross Value", "Value",
                "Slippage", "Brokerage", "STT", "Stamp Duty", "GST", "Exchange Charges",
                "Commission", "Transaction Cost", "Net Cash Movement", "Net Value",
                "Position Before", "Position After", "Cash Before", "Cash After",
                "Cash", "Remaining Cash", "Realized PnL", "Cumulative Realized PnL",
                "Unrealized PnL", "Portfolio Value", "Portfolio Equity",
                "Order Status", "Rejection Reason", "Reason", "Order ID", "Fill ID"
            ]
            df = pd.DataFrame(columns=cols)
            df.index.name = "Date"
            return df

        df = pd.DataFrame(self.records)
        df.set_index("Date", inplace=True)
        return df
