"""
result.py

BacktestResult container object holding complete simulation output.
Serves as canonical interface for UI, AI agents, report generators, and LLMs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
from .config import BacktestConfig
from .order import Order
from .fill import Fill


@dataclass
class BacktestResult:
    """
    Clean result structure for executed backtest simulations.

    Attributes
    ----------
    portfolio_history : pd.DataFrame
        Bar-by-bar time series of cash, holdings, portfolio value, P&L, exposure.
    orders : List[Order]
        Complete list of generated orders.
    fills : List[Fill]
        Complete list of executed fills.
    trade_log_df : pd.DataFrame
        Canonical trade log DataFrame.
    metrics : Dict[str, Any]
        Deterministic quantitative performance KPIs.
    benchmark : Dict[str, Any]
        Benchmark comparison results.
    execution_statistics : Dict[str, Any]
        Detailed execution layer statistics (slippage total, commission total, etc.).
    configuration : BacktestConfig
        Configuration object used to run the simulation.
    execution_trace : List[str]
        Debug/trace logs captured per bar/event.
    """
    portfolio_history: pd.DataFrame
    orders: List[Order] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)
    trade_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, Any] = field(default_factory=dict)
    benchmark: Dict[str, Any] = field(default_factory=dict)
    execution_statistics: Dict[str, Any] = field(default_factory=dict)
    configuration: Optional[BacktestConfig] = None
    execution_trace: List[str] = field(default_factory=list)

    def get_metrics(self) -> Dict[str, Any]:
        """Return deterministic metric KPIs."""
        return self.metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade log DataFrame."""
        return self.trade_log_df

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Return execution friction statistics."""
        return self.execution_statistics

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Return final portfolio state snapshot."""
        if self.portfolio_history is not None and not self.portfolio_history.empty:
            last_row = self.portfolio_history.iloc[-1]
            return {
                "Cash": last_row.get("Cash", 0.0),
                "Holdings Value": last_row.get("Holdings Value", 0.0),
                "Portfolio Value": last_row.get("Portfolio Value", 0.0),
                "Realized PnL": last_row.get("Realized PnL", 0.0),
                "Unrealized PnL": last_row.get("Unrealized PnL", 0.0),
                "Total PnL": last_row.get("Total PnL", 0.0),
                "Exposure": last_row.get("Exposure", 0.0)
            }
        return {}

    def get_trace_log(self) -> List[str]:
        return self.execution_trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "execution_statistics": self.execution_statistics,
            "portfolio_state": self.get_portfolio_state(),
            "config": self.configuration.to_dict() if self.configuration else {}
        }

