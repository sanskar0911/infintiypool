"""
config.py

Configuration schema and settings dataclass for the QUANTLAB Backtesting Engine.
Supports serialization to/from dictionary and JSON for reproducibility.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum


class ExecutionTiming(str, Enum):
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"
    SAME_BAR_CLOSE = "SAME_BAR_CLOSE"
    NEXT_BAR_CLOSE = "NEXT_BAR_CLOSE"


class InsufficientCashPolicy(str, Enum):
    REDUCE_QUANTITY = "REDUCE_QUANTITY"
    REJECT = "REJECT"


@dataclass
class BacktestConfig:
    """
    Backtest Engine Configuration.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio balance in INR (default: 100,000.0).
    position_size : float
        Fraction of available capital allocated per position (0.0 to 1.0, default: 0.20).
    slippage_bps : float
        Slippage penalty in basis points (10 bps = 0.10%, default: 10.0).
    commission_bps : float
        Transaction fee / brokerage in basis points (5 bps = 0.05%, default: 5.0).
    execution_timing : ExecutionTiming
        Bar execution timing mode (default: NEXT_BAR_OPEN).
    insufficient_cash_policy : InsufficientCashPolicy
        Action when available cash is insufficient for target order (default: REDUCE_QUANTITY).
    allow_short_selling : bool
        Whether short selling is permitted (default: False).
    stop_loss : float, optional
        Stop loss percentage threshold (e.g. 0.05 for 5%).
    take_profit : float, optional
        Take profit percentage threshold (e.g. 0.10 for 10%).
    price_column : str
        Default price column for valuation (default: "Close").
    """
    initial_capital: float = 100000.0
    position_size: float = 0.20
    slippage_bps: float = 10.0
    commission_bps: float = 5.0
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    insufficient_cash_policy: InsufficientCashPolicy = InsufficientCashPolicy.REDUCE_QUANTITY
    allow_short_selling: bool = False
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    price_column: str = "Close"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["execution_timing"] = str(self.execution_timing)
        d["insufficient_cash_policy"] = str(self.insufficient_cash_policy)
        return d

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """Reconstruct configuration from dictionary."""
        d = data.copy()
        if "execution_timing" in d and isinstance(d["execution_timing"], str):
            d["execution_timing"] = ExecutionTiming(d["execution_timing"])
        if "insufficient_cash_policy" in d and isinstance(d["insufficient_cash_policy"], str):
            d["insufficient_cash_policy"] = InsufficientCashPolicy(d["insufficient_cash_policy"])
        
        # Handle conversion from rate (0.001) to bps (10 bps) if passed as txn_cost_rate or slippage_rate
        if "txn_cost_rate" in d and "commission_bps" not in d:
            d["commission_bps"] = d.pop("txn_cost_rate") * 10000.0
        if "slippage_rate" in d and "slippage_bps" not in d:
            d["slippage_bps"] = d.pop("slippage_rate") * 10000.0

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_kwargs = {k: v for k, v in d.items() if k in valid_keys}
        extra = {k: v for k, v in d.items() if k not in valid_keys}
        if extra:
            filtered_kwargs["extra_params"] = extra

        return cls(**filtered_kwargs)
