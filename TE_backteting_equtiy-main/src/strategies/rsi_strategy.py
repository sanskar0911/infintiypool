"""
rsi_strategy.py

Relative Strength Index (RSI) Strategy implementation.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure path imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.base import BaseStrategy
from utils import daily_returns


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy.
    Generates BUY signal when RSI crosses above oversold threshold (default 30),
    and SELL signal when RSI crosses below overbought threshold (default 70).
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        price_col: str = "Adj Close"
    ):
        super().__init__(name=f"RSI Strategy ({period}, {oversold}/{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.price_col = price_col

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.price_col if self.price_col in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        df_ind = df.copy()

        delta = df_ind[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df_ind["RSI"] = rsi.fillna(50)

        if "Returns" not in df_ind.columns:
            df_ind["Returns"] = daily_returns(df_ind, price_col=col)

        return df_ind

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sig = self.calculate_indicators(df)

        rsi_curr = df_sig["RSI"]
        rsi_prev = df_sig["RSI"].shift(1)

        df_sig["Signal"] = 0

        # BUY when RSI crosses above oversold threshold
        buy_condition = (rsi_prev <= self.oversold) & (rsi_curr > self.oversold)
        # SELL when RSI crosses below overbought threshold
        sell_condition = (rsi_prev >= self.overbought) & (rsi_curr < self.overbought)

        df_sig.loc[buy_condition, "Signal"] = 1
        df_sig.loc[sell_condition, "Signal"] = -1

        position_target = pd.Series(index=df_sig.index, dtype=float)
        position_target[df_sig["Signal"] == 1] = 1.0
        position_target[df_sig["Signal"] == -1] = 0.0

        df_sig["Position"] = position_target.ffill().fillna(0.0).astype(int)

        return df_sig
