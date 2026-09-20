"""
backtester.py

Backward-compatibility wrapper for QUANTLAB Backtesting Engine.
Re-exports modular components from src.engine.
"""

import os
import sys
from typing import Tuple, Optional
import pandas as pd

# Path setup to ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine import (
    BacktestEngine,
    BacktestConfig,
    ExecutionTiming,
    InsufficientCashPolicy,
    Order,
    Fill,
    PortfolioManager,
    BacktestResult,
    run_backtest as run_engine_backtest
)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    txn_cost_rate: float = 0.001,      # 0.1% commission
    slippage_rate: float = 0.0005,     # 0.05% slippage
    position_size: float = 1.0,        # Capital allocation fraction
    stop_loss: Optional[float] = None, # Stop loss fraction
    take_profit: Optional[float] = None, # Take profit fraction
    execution_timing: str = "NEXT_BAR_OPEN",
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest simulation via upgraded src.engine BacktestEngine.
    Maintains 100% backward compatibility for existing callers.
    """
    return run_engine_backtest(
        df=df,
        initial_capital=initial_capital,
        txn_cost_rate=txn_cost_rate,
        slippage_rate=slippage_rate,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        execution_timing=execution_timing,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing src/backtester.py backward-compatible wrapper...")
    try:
        from strategy import generate_signals

        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "RELIANCE.csv")
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path, index_col="Date")
            df_sample.index = pd.to_datetime(df_sample.index)

            df_signals = generate_signals(df_sample)
            portfolio_df, trade_log_df = run_backtest(
                df_signals,
                position_size=0.2,
                stop_loss=0.05,
                take_profit=0.10
            )
            print(f"Engine test successful! Trades executed: {len(trade_log_df)}")
            print(trade_log_df.head(4))
    except Exception as e:
        print(f"Self-test notification: {e}")
