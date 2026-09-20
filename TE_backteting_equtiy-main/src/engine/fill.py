"""
fill.py

Explicit Fill object representing the canonical record of an executed trade.
"""

from dataclasses import dataclass, field
import uuid
import pandas as pd
from .order import OrderSide


@dataclass
class Fill:
    """
    Represents an executed order fill.

    Attributes
    ----------
    timestamp : pd.Timestamp
        Actual timestamp of execution bar.
    symbol : str
        Stock ticker symbol.
    side : OrderSide
        BUY or SELL.
    quantity : int
        Number of shares filled.
    reference_price : float
        Base price before slippage.
    fill_price : float
        Actual execution price after slippage.
    slippage_cost : float
        Total monetary impact of slippage (abs(fill_price - reference_price) * quantity).
    transaction_cost : float
        Total brokerage and fees.
    gross_value : float
        quantity * fill_price.
    net_value : float
        For BUY: gross_value + transaction_cost (cash outflow).
        For SELL: gross_value - transaction_cost (cash inflow).
    order_id : str
        Parent order ID.
    fill_id : str
        Unique fill ID.
    reason : str
        Execution trigger reason.
    """
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: int
    reference_price: float
    fill_price: float
    slippage_cost: float
    transaction_cost: float
    gross_value: float
    net_value: float
    order_id: str
    fill_id: str = field(default_factory=lambda: f"FIL-{uuid.uuid4().hex[:8].upper()}")
    reason: str = "Execution"

    def to_dict(self) -> dict:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "timestamp": str(self.timestamp),
            "symbol": self.symbol,
            "side": str(self.side),
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "fill_price": self.fill_price,
            "slippage_cost": self.slippage_cost,
            "transaction_cost": self.transaction_cost,
            "gross_value": self.gross_value,
            "net_value": self.net_value,
            "reason": self.reason
        }
