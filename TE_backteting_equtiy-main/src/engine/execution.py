"""
execution.py

Execution and Slippage Model for QUANTLAB Backtesting Engine.
Enforces strict look-ahead safety via configurable execution timing modes
and applies adverse price slippage.
"""

from typing import Tuple, Optional
import pandas as pd
from .config import BacktestConfig, ExecutionTiming
from .order import Order, OrderSide


class SlippageModel:
    """
    Applies adverse slippage based on basis points (10 bps = 0.10%).

    BUY: Fill price increases above base price.
    SELL: Fill price decreases below base price.
    """

    def __init__(self, slippage_bps: float = 10.0):
        self.slippage_bps = float(slippage_bps)
        self.rate = self.slippage_bps / 10000.0

    def calculate_fill_price(self, base_price: float, side: OrderSide) -> Tuple[float, float]:
        """
        Calculate fill price and absolute slippage per share.

        Returns
        -------
        Tuple[float, float]
            (fill_price, slippage_per_share)
        """
        if side == OrderSide.BUY:
            fill_price = base_price * (1.0 + self.rate)
            slippage_per_share = fill_price - base_price
        else:
            fill_price = base_price * (1.0 - self.rate)
            slippage_per_share = base_price - fill_price

        return fill_price, max(0.0, slippage_per_share)


class ExecutionModel:
    """
    Orchestrates execution timing and base price resolution without look-ahead bias.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.slippage_model = SlippageModel(config.slippage_bps)

    def resolve_execution_price(
        self,
        order: Order,
        current_bar_idx: int,
        df: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[pd.Timestamp], str]:
        """
        Determine base price and timestamp for execution based on timing configuration.

        NEXT_BAR_OPEN: Order generated at T (signal bar), executed at T+1 Open.
        SAME_BAR_CLOSE: Order generated at T, executed at T Close (legacy/unrecommended mode).
        NEXT_BAR_CLOSE: Order generated at T, executed at T+1 Close.

        Returns
        -------
        Tuple[Optional[float], Optional[pd.Timestamp], str]
            (base_price, execution_timestamp, timing_note)
        """
        signal_bar = order.signal_bar_idx
        n_bars = len(df)

        if self.config.execution_timing == ExecutionTiming.NEXT_BAR_OPEN:
            exec_bar = signal_bar + 1
            if exec_bar >= n_bars:
                return None, None, "Pending next bar open (end of dataset)"
            
            # Check if Open column exists
            price_col = "Open" if "Open" in df.columns else self.config.price_column
            base_price = float(df[price_col].iloc[exec_bar])
            exec_time = df.index[exec_bar]
            return base_price, exec_time, "Next Bar Open"

        elif self.config.execution_timing == ExecutionTiming.NEXT_BAR_CLOSE:
            exec_bar = signal_bar + 1
            if exec_bar >= n_bars:
                return None, None, "Pending next bar close (end of dataset)"

            price_col = self.config.price_column
            base_price = float(df[price_col].iloc[exec_bar])
            exec_time = df.index[exec_bar]
            return base_price, exec_time, "Next Bar Close"

        elif self.config.execution_timing == ExecutionTiming.SAME_BAR_CLOSE:
            exec_bar = signal_bar
            price_col = self.config.price_column
            base_price = float(df[price_col].iloc[exec_bar])
            exec_time = df.index[exec_bar]
            return base_price, exec_time, "Same Bar Close"

        else:
            raise ValueError(f"Unsupported execution timing: {self.config.execution_timing}")
