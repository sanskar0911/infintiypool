"""
QUANTLAB Modular Backtesting Engine Package.
Order -> Execution -> Fill -> Portfolio State -> Metrics
"""

from .config import BacktestConfig, ExecutionTiming, InsufficientCashPolicy
from .order import Order, OrderSide, OrderType, OrderStatus
from .position_sizing import PositionSizer
from .execution import ExecutionModel, SlippageModel
from .cost_model import TransactionCostModel
from .fill import Fill
from .portfolio import PortfolioManager, PositionTracker
from .trade_log import TradeLogger
from .validation import DataValidator, DataValidationError
from .result import BacktestResult
from .engine import BacktestEngine, run_backtest

__all__ = [
    "BacktestConfig",
    "ExecutionTiming",
    "InsufficientCashPolicy",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionSizer",
    "ExecutionModel",
    "SlippageModel",
    "TransactionCostModel",
    "Fill",
    "PortfolioManager",
    "PositionTracker",
    "TradeLogger",
    "DataValidator",
    "DataValidationError",
    "BacktestResult",
    "BacktestEngine",
    "run_backtest"
]
