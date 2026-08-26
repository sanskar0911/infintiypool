"""
sma_crossover.py

Simple Moving Average (SMA) Crossover Strategy implementation.
"""

import os
import sys
import pandas as pd
from typing import Optional

# Ensure path imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.base import BaseStrategy
from utils import moving_average, daily_returns


class SMACrossoverStrategy(BaseStrategy):
    """
    SMA Crossover Strategy.
    Generates BUY signal when short-term SMA crosses above long-term SMA,
    and SELL signal when short-term SMA crosses below long-term SMA.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50, price_col: str = "Adj Close"):
        super().__init__(name=f"SMA Crossover ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window
        self.price_col = price_col

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.price_col if self.price_col in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        df_ind = df.copy()

        df_ind[f"SMA{self.short_window}"] = moving_average(df_ind, window=self.short_window, price_col=col)
        df_ind[f"SMA{self.long_window}"] = moving_average(df_ind, window=self.long_window, price_col=col)
        
        # Legacy column naming support
        df_ind["SMA20"] = moving_average(df_ind, window=20, price_col=col)
        df_ind["SMA50"] = moving_average(df_ind, window=50, price_col=col)
        df_ind["SMA200"] = moving_average(df_ind, window=200, price_col=col)

        if "Returns" not in df_ind.columns:
            df_ind["Returns"] = daily_returns(df_ind, price_col=col)

        return df_ind

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sig = self.calculate_indicators(df)
        
        short_col = f"SMA{self.short_window}"
        long_col = f"SMA{self.long_window}"

        sma_short_curr = df_sig[short_col]
        sma_long_curr = df_sig[long_col]
        sma_short_prev = df_sig[short_col].shift(1)
        sma_long_prev = df_sig[long_col].shift(1)

        df_sig["Signal"] = 0

        buy_condition = (sma_short_prev <= sma_long_prev) & (sma_short_curr > sma_long_curr)
        sell_condition = (sma_short_prev >= sma_long_prev) & (sma_short_curr < sma_long_curr)

        df_sig.loc[buy_condition, "Signal"] = 1
        df_sig.loc[sell_condition, "Signal"] = -1

        position_target = pd.Series(index=df_sig.index, dtype=float)
        position_target[df_sig["Signal"] == 1] = 1.0
        position_target[df_sig["Signal"] == -1] = 0.0

        df_sig["Position"] = position_target.ffill().fillna(0.0).astype(int)

        return df_sig
