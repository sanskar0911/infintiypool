"""
ema_crossover.py

Exponential Moving Average (EMA) Crossover Strategy implementation.
"""

import os
import sys
import pandas as pd

# Ensure path imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.base import BaseStrategy
from utils import daily_returns


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.
    Generates BUY signal when short-term EMA crosses above long-term EMA,
    and SELL signal when short-term EMA crosses below long-term EMA.
    """

    def __init__(self, short_window: int = 12, long_window: int = 26, price_col: str = "Adj Close"):
        super().__init__(name=f"EMA Crossover ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window
        self.price_col = price_col

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.price_col if self.price_col in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        df_ind = df.copy()

        df_ind[f"EMA{self.short_window}"] = df_ind[col].ewm(span=self.short_window, adjust=False).mean()
        df_ind[f"EMA{self.long_window}"] = df_ind[col].ewm(span=self.long_window, adjust=False).mean()

        if "Returns" not in df_ind.columns:
            df_ind["Returns"] = daily_returns(df_ind, price_col=col)

        return df_ind

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sig = self.calculate_indicators(df)

        short_col = f"EMA{self.short_window}"
        long_col = f"EMA{self.long_window}"

        ema_short_curr = df_sig[short_col]
        ema_long_curr = df_sig[long_col]
        ema_short_prev = df_sig[short_col].shift(1)
        ema_long_prev = df_sig[long_col].shift(1)

        df_sig["Signal"] = 0

        buy_condition = (ema_short_prev <= ema_long_prev) & (ema_short_curr > ema_long_curr)
        sell_condition = (ema_short_prev >= ema_long_prev) & (ema_short_curr < ema_long_curr)

        df_sig.loc[buy_condition, "Signal"] = 1
        df_sig.loc[sell_condition, "Signal"] = -1

        position_target = pd.Series(index=df_sig.index, dtype=float)
        position_target[df_sig["Signal"] == 1] = 1.0
        position_target[df_sig["Signal"] == -1] = 0.0

        df_sig["Position"] = position_target.ffill().fillna(0.0).astype(int)

        return df_sig
