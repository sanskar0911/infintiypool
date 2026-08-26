"""
strategy.py

Facade module preserving original function signatures for strategy calculations.
Delegates internally to the modular Strategy Library under src/strategies/.
"""

import os
import sys
import pandas as pd
from typing import Optional

# Ensure local path imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategies.factory import StrategyFactory
from strategies.sma_crossover import SMACrossoverStrategy


def calculate_indicators(df: pd.DataFrame, strategy_name: str = "SMA", **kwargs) -> pd.DataFrame:
    """
    Calculate technical indicators for the backtest.
    Backward-compatible wrapper around StrategyFactory.
    """
    strategy = StrategyFactory.get_strategy(strategy_name, **kwargs)
    return strategy.calculate_indicators(df)


def generate_signals(df: pd.DataFrame, strategy_name: str = "SMA", **kwargs) -> pd.DataFrame:
    """
    Generate trading signals using the specified strategy.
    Backward-compatible wrapper around StrategyFactory.
    """
    strategy = StrategyFactory.get_strategy(strategy_name, **kwargs)
    return strategy.generate_signals(df)


if __name__ == "__main__":
    print("Testing strategy.py facade backward-compatibility...")
    try:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "RELIANCE.csv")
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path, index_col="Date")
            df_sample.index = pd.to_datetime(df_sample.index)
            
            # Test SMA default
            df_signals_sma = generate_signals(df_sample, strategy_name="SMA")
            print(f"SMA signals shape: {df_signals_sma.shape}")
            
            # Test EMA
            df_signals_ema = generate_signals(df_sample, strategy_name="EMA")
            print(f"EMA signals shape: {df_signals_ema.shape}")
            
            # Test RSI
            df_signals_rsi = generate_signals(df_sample, strategy_name="RSI")
            print(f"RSI signals shape: {df_signals_rsi.shape}")
    except Exception as e:
        print(f"Self-test error: {e}")
