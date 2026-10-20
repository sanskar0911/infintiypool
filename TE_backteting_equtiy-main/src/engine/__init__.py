"""
QUANTLAB Modular Backtesting Engine Package v1.1.
Order -> Execution -> Fill -> Portfolio State -> Metrics
"""

from .config import (
    BacktestConfig,
    ExecutionTiming,
    InsufficientCashPolicy,
    SizingBasis,
    get_default_baseline_config
)
from .order import Order, OrderSide, OrderType, OrderStatus
from .position_sizing import PositionSizer
from .execution import ExecutionModel, SlippageModel
from .cost_model import TransactionCostModel, CostSchedule
from .fill import Fill
from .portfolio import PortfolioManager, PositionTracker
from .trade_log import TradeLogger
from .validation import DataValidator, DataValidationError
from .result import BacktestResult
from .engine import BacktestEngine, run_backtest
from .signal_ingestion import run_signal_file_backtest, load_and_standardize_signals
from .llm_auditor import run_quantitative_llm_audit
from .mt5_adapter import MT5ReconciliationAdapter, MT5SymbolSpecification

__all__ = [
    "BacktestConfig",
    "ExecutionTiming",
    "InsufficientCashPolicy",
    "SizingBasis",
    "get_default_baseline_config",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionSizer",
    "ExecutionModel",
    "SlippageModel",
    "TransactionCostModel",
    "CostSchedule",
    "Fill",
    "PortfolioManager",
    "PositionTracker",
    "TradeLogger",
    "DataValidator",
    "DataValidationError",
    "BacktestResult",
    "BacktestEngine",
    "run_backtest",
    "run_signal_file_backtest",
    "load_and_standardize_signals",
    "run_quantitative_llm_audit",
    "MT5ReconciliationAdapter",
    "MT5SymbolSpecification"
]
