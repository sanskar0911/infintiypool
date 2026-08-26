"""
factory.py

Strategy Factory pattern implementation for dynamic strategy instantiation.
"""

import os
import sys
from typing import Dict, Type, List, Any

# Ensure path imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.base import BaseStrategy
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy


class StrategyFactory:
    """Factory class to manage and instantiate trading strategies."""

    _strategies: Dict[str, Type[BaseStrategy]] = {
        "SMA": SMACrossoverStrategy,
        "EMA": EMACrossoverStrategy,
        "RSI": RSIStrategy,
    }

    @classmethod
    def get_strategy(cls, name: str, **kwargs: Any) -> BaseStrategy:
        """
        Instantiate a strategy by name.

        Parameters
        ----------
        name : str
            Strategy name identifier ('SMA', 'EMA', or 'RSI').
        kwargs : dict
            Custom keyword arguments for the strategy constructor.

        Returns
        -------
        BaseStrategy
            An initialized concrete strategy instance.
        """
        strategy_key = name.upper().strip()
        if strategy_key not in cls._strategies:
            raise ValueError(
                f"Unknown strategy '{name}'. Available strategies: {list(cls._strategies.keys())}"
            )
        
        return cls._strategies[strategy_key](**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Return list of available strategy names."""
        return list(cls._strategies.keys())


if __name__ == "__main__":
    print("Testing StrategyFactory...")
    print(f"Available strategies: {StrategyFactory.list_strategies()}")
    sma_strat = StrategyFactory.get_strategy("SMA", short_window=20, long_window=50)
    print(f"Instantiated: {sma_strat.name}")
