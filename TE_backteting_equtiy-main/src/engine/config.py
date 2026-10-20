"""
config.py

Configuration schema and settings dataclass for the QUANTLAB Backtesting Engine v1.1.
Supports serialization to/from YAML and JSON for complete reproducibility.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path


class ExecutionTiming(str, Enum):
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"
    SAME_BAR_CLOSE = "SAME_BAR_CLOSE"
    NEXT_BAR_CLOSE = "NEXT_BAR_CLOSE"


class InsufficientCashPolicy(str, Enum):
    REDUCE_QUANTITY = "REDUCE_QUANTITY"
    REJECT = "REJECT"


class SizingBasis(str, Enum):
    EQUITY = "equity"
    CASH = "cash"


@dataclass
class BacktestConfig:
    """
    QUANTLAB Unified Backtest Engine Configuration v1.1.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio balance in INR (default: ₹10,000,000 / ₹1 Crore).
    position_size : float
        Target allocation fraction per position (default: 0.05 / 5.0% of current equity).
    sizing_basis : SizingBasis or str
        "equity" (5% of current portfolio equity) or "cash" (5% of remaining cash).
    slippage_bps : float
        Adverse slippage penalty in basis points (default: 5.0 bps = 0.05%, s = 0.0005).
    brokerage_bps : float
        Baseline brokerage per side in basis points (default: 3.0 bps = 0.03%).
    commission_bps : float, optional
        Backward-compatibility alias for brokerage_bps.
    stt_rate_sell : float
        Securities Transaction Tax on sell-side (default: 0.0010 = 0.10%).
    stamp_duty_rate_buy : float
        Stamp Duty on buy-side (default: 0.00015 = 0.015%).
    gst_rate : float
        GST on brokerage + exchange charges (default: 0.18 = 18%).
    exchange_charge_rate : float
        Exchange transaction charges (default: 0.0 = 0.0%).
    sebi_charge_rate : float
        SEBI turnover charges (default: 0.0 = 0.0%).
    execution_timing : ExecutionTiming
        Bar execution timing mode (default: NEXT_BAR_OPEN).
    insufficient_cash_policy : InsufficientCashPolicy
        Action when available cash is insufficient (default: REJECT).
    allow_short_selling : bool
        Whether short selling is permitted (default: False, long-only).
    stop_loss : float, optional
        Stop loss percentage threshold (e.g. 0.05 for 5%).
    take_profit : float, optional
        Take profit percentage threshold (e.g. 0.10 for 10%).
    risk_free_rate : float
        Annualized risk-free rate for Sharpe/Sortino (default: 0.0 = 0%).
    annualization_days : int
        Trading days per year (default: 252).
    terminal_force_close : bool
        Whether to force-close open positions at final price (default: True).
    price_column : str
        Default price column for valuation (default: "Close").
    """
    initial_capital: float = 10000000.0
    position_size: float = 0.05
    sizing_basis: str = "equity"
    slippage_bps: float = 5.0           # 0.05% adverse per side
    brokerage_bps: float = 3.0          # 0.03% baseline per side
    commission_bps: Optional[float] = None # Backward compatibility alias
    stt_rate_sell: float = 0.0010       # 0.10% sell STT
    stamp_duty_rate_buy: float = 0.00015 # 0.015% buy Stamp Duty
    gst_rate: float = 0.18              # 18% GST
    exchange_charge_rate: float = 0.0   # 0% baseline
    sebi_charge_rate: float = 0.0       # 0% baseline
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    insufficient_cash_policy: InsufficientCashPolicy = InsufficientCashPolicy.REJECT
    allow_short_selling: bool = False
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_free_rate: float = 0.0
    annualization_days: int = 252
    terminal_force_close: bool = True
    price_column: str = "Close"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.commission_bps is not None:
            self.brokerage_bps = float(self.commission_bps)
        else:
            self.commission_bps = float(self.brokerage_bps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["execution_timing"] = str(self.execution_timing)
        d["insufficient_cash_policy"] = str(self.insufficient_cash_policy)
        d["commission_bps"] = self.brokerage_bps
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
        
        # Backward compatibility mappings
        if "commission_bps" in d and "brokerage_bps" not in d:
            d["brokerage_bps"] = float(d["commission_bps"])
        if "txn_cost_rate" in d:
            d["brokerage_bps"] = float(d.pop("txn_cost_rate")) * 10000.0
            d["commission_bps"] = d["brokerage_bps"]
        if "slippage_rate" in d and "slippage_bps" not in d:
            d["slippage_bps"] = float(d.pop("slippage_rate")) * 10000.0

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_kwargs = {k: v for k, v in d.items() if k in valid_keys}
        extra = {k: v for k, v in d.items() if k not in valid_keys}
        if extra:
            filtered_kwargs["extra_params"] = extra

        return cls(**filtered_kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BacktestConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        flat: Dict[str, Any] = {}
        if "portfolio" in raw:
            p = raw["portfolio"]
            flat["initial_capital"] = float(p.get("initial_capital", 10000000.0))
            flat["allow_short_selling"] = bool(p.get("allow_short_selling", False))
            flat["terminal_force_close"] = (p.get("terminal_policy") == "force_close")

        if "position_sizing" in raw:
            ps = raw["position_sizing"]
            flat["position_size"] = float(ps.get("target_allocation_fraction", 0.05))
            flat["sizing_basis"] = str(ps.get("sizing_basis", "equity"))
            policy_str = str(ps.get("insufficient_cash_policy", "REJECT"))
            flat["insufficient_cash_policy"] = InsufficientCashPolicy(policy_str)

        if "execution" in raw:
            ex = raw["execution"]
            timing_str = str(ex.get("timing", "NEXT_BAR_OPEN"))
            flat["execution_timing"] = ExecutionTiming(timing_str)
            flat["slippage_bps"] = float(ex.get("slippage_rate", 0.0005)) * 10000.0

        if "transaction_costs" in raw:
            tc = raw["transaction_costs"]
            flat["brokerage_bps"] = round(float(tc.get("brokerage_rate", 0.0003)) * 10000.0, 4)
            flat["commission_bps"] = flat["brokerage_bps"]
            flat["stt_rate_sell"] = float(tc.get("stt_rate_sell", 0.0010))
            flat["stamp_duty_rate_buy"] = float(tc.get("stamp_duty_rate_buy", 0.00015))
            flat["gst_rate"] = float(tc.get("gst_rate", 0.18))
            flat["exchange_charge_rate"] = float(tc.get("exchange_charge_rate", 0.0))
            flat["sebi_charge_rate"] = float(tc.get("sebi_charge_rate", 0.0))

        if "performance_metrics" in raw:
            pm = raw["performance_metrics"]
            flat["annualization_days"] = int(pm.get("annualization_days", 252))
            flat["risk_free_rate"] = float(pm.get("risk_free_rate", 0.0))

        # Merge with top-level fields
        for k, v in raw.items():
            if k not in ["portfolio", "position_sizing", "execution", "transaction_costs", "performance_metrics", "order_sequencing", "audit_and_reproducibility"]:
                flat[k] = v

        return cls.from_dict(flat)


def get_default_baseline_config() -> BacktestConfig:
    """Return default QUANTLAB v1.1 baseline configuration."""
    yaml_candidates = [
        Path(__file__).resolve().parent.parent.parent / "config" / "backtest_baseline.yaml",
        Path(__file__).resolve().parent.parent / "config" / "backtest_baseline.yaml",
        Path("config/backtest_baseline.yaml"),
    ]
    for c in yaml_candidates:
        if c.exists():
            try:
                return BacktestConfig.from_yaml(str(c))
            except Exception:
                pass
    return BacktestConfig()
