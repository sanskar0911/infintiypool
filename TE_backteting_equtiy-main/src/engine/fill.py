"""
fill.py

Explicit Fill object representing the canonical execution record for QUANTLAB v1.1.
Contains detailed breakdown of Indian equity transaction fees and adverse slippage.
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
        Number of integer shares filled.
    reference_price : float
        Base price before slippage.
    fill_price : float
        Actual execution price after adverse slippage.
    slippage_cost : float
        Total adverse slippage amount (abs(fill_price - reference_price) * quantity).
    brokerage : float
        Brokerage fee in INR.
    stt : float
        Securities Transaction Tax in INR.
    stamp_duty : float
        Stamp Duty in INR.
    gst : float
        Goods and Services Tax in INR.
    exchange_charges : float
        Exchange transaction charges in INR.
    transaction_cost : float
        Total statutory transaction costs (brokerage + STT + stamp_duty + GST + exchange).
    gross_value : float
        quantity * fill_price.
    net_value : float
        For BUY: gross_value + transaction_cost (total cash outflow).
        For SELL: gross_value - transaction_cost (net cash inflow).
    order_id : str
        Parent order ID.
    fill_id : str
        Unique fill identifier.
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
    brokerage: float = 0.0
    stt: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    exchange_charges: float = 0.0
    transaction_cost: float = 0.0
    gross_value: float = 0.0
    net_value: float = 0.0
    order_id: str = ""
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
            "brokerage": self.brokerage,
            "stt": self.stt,
            "stamp_duty": self.stamp_duty,
            "gst": self.gst,
            "exchange_charges": self.exchange_charges,
            "transaction_cost": self.transaction_cost,
            "gross_value": self.gross_value,
            "net_value": self.net_value,
            "reason": self.reason
        }
