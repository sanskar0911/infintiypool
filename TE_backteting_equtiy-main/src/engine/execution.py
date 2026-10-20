"""
execution.py

Execution and Slippage Engine for QUANTLAB v1.1.
Enforces strict look-ahead safety via completed-bar signal confirmation and applies
adverse price slippage:
- Adverse fixed-percentage slippage: s = 0.0005 (0.05% adverse per side)
- Buys: P_buy = P_reference * (1 + s)
- Sells: P_sell = P_reference * (1 - s)
- Bid/Ask support: Ask as buy reference, Bid as sell reference when available.
- Mode A: Completed-bar signals confirmed at close, executed at next available bar quote.
- Mode B: Intrabar or tick-time execution using current observable quote.
"""

from typing import Tuple, Optional
import pandas as pd
from .config import BacktestConfig, ExecutionTiming
from .order import Order, OrderSide


class SlippageModel:
    """
    Adverse Slippage Model.
    Default: s = 0.0005 (0.05% adverse per side).
    """

    def __init__(self, slippage_bps: float = 5.0):
        self.slippage_bps = float(slippage_bps)
        self.rate = self.slippage_bps / 10000.0  # 5 bps -> 0.0005

    def calculate_fill_price(self, base_price: float, side: OrderSide) -> Tuple[float, float]:
        """
        Calculate fill price with adverse slippage and slippage cost per share.

        Parameters
        ----------
        base_price : float
            Executable reference price.
        side : OrderSide
            BUY or SELL.

        Returns
        -------
        Tuple[float, float]
            (fill_price, slippage_per_share)
        """
        is_buy = (side == OrderSide.BUY or str(side).upper() == "BUY")

        if is_buy:
            fill_price = base_price * (1.0 + self.rate)
            slippage_per_share = fill_price - base_price
        else:
            fill_price = base_price * (1.0 - self.rate)
            slippage_per_share = base_price - fill_price

        return fill_price, max(0.0, slippage_per_share)


class ExecutionModel:
    """
    Resolves execution reference price without lookahead bias.
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
        Determine base price and execution timestamp based on execution timing configuration.

        Mode A:
        - NEXT_BAR_OPEN: Signal generated at T (completed bar close), executed at T+1 Open.
        - NEXT_BAR_CLOSE: Signal generated at T, executed at T+1 Close.
        - SAME_BAR_CLOSE: Completed bar close execution (only if explicitly configured).

        Returns
        -------
        Tuple[Optional[float], Optional[pd.Timestamp], str]
            (base_price, execution_timestamp, timing_note)
        """
        signal_bar = order.signal_bar_idx
        n_bars = len(df)
        is_buy = order.is_buy()

        if self.config.execution_timing == ExecutionTiming.NEXT_BAR_OPEN:
            exec_bar = signal_bar + 1
            if exec_bar >= n_bars:
                return None, None, "Pending next bar open (end of dataset)"

            # Spread-aware pricing if Ask / Bid columns are present
            if is_buy and "Ask" in df.columns:
                base_price = float(df["Ask"].iloc[exec_bar])
            elif not is_buy and "Bid" in df.columns:
                base_price = float(df["Bid"].iloc[exec_bar])
            elif "Open" in df.columns:
                base_price = float(df["Open"].iloc[exec_bar])
            else:
                base_price = float(df[self.config.price_column].iloc[exec_bar])

            exec_time = df.index[exec_bar]
            return base_price, exec_time, "Next Bar Open"

        elif self.config.execution_timing == ExecutionTiming.NEXT_BAR_CLOSE:
            exec_bar = signal_bar + 1
            if exec_bar >= n_bars:
                return None, None, "Pending next bar close (end of dataset)"

            if is_buy and "Ask" in df.columns:
                base_price = float(df["Ask"].iloc[exec_bar])
            elif not is_buy and "Bid" in df.columns:
                base_price = float(df["Bid"].iloc[exec_bar])
            else:
                base_price = float(df[self.config.price_column].iloc[exec_bar])

            exec_time = df.index[exec_bar]
            return base_price, exec_time, "Next Bar Close"

        elif self.config.execution_timing == ExecutionTiming.SAME_BAR_CLOSE:
            exec_bar = signal_bar
            if is_buy and "Ask" in df.columns:
                base_price = float(df["Ask"].iloc[exec_bar])
            elif not is_buy and "Bid" in df.columns:
                base_price = float(df["Bid"].iloc[exec_bar])
            else:
                base_price = float(df[self.config.price_column].iloc[exec_bar])

            exec_time = df.index[exec_bar]
            return base_price, exec_time, "Same Bar Close"

        return None, None, "Unrecognized execution timing"
