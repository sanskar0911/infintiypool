"""
base.py

Abstract base class for all trading strategy implementations.
Enforces a standardized contract for indicator calculations and signal generation.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base strategy class defining mandatory interface methods."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators required for the strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned price dataframe (containing 'Adj Close' or 'Close').

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with technical indicator columns.
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals (1=BUY, -1=SELL, 0=HOLD) and positions (1=Long, 0=Cash).

        Parameters
        ----------
        df : pd.DataFrame
            Price dataframe.

        Returns
        -------
        pd.DataFrame
            DataFrame containing 'Signal' and 'Position' columns.
        """
        pass
