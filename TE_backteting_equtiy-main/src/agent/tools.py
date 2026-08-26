"""
tools.py

LangGraph Executable Tools for Backtesting, Performance Metrics & LLM Analysis.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any

# Ensure path imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import load_stock_data
from strategy import generate_signals
from backtester import run_backtest
from benchmark import compare_strategy_vs_benchmark, load_benchmark_data
from metrics import calculate_metrics
from llm.analyzer import LLMStrategyAnalyzer


def run_data_loader_tool(ticker: str) -> pd.DataFrame:
    """Load historical stock data for given ticker."""
    data_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Stock data file for {ticker} not found at {data_path}")

    df = pd.read_csv(data_path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def run_backtest_tool(
    df: pd.DataFrame,
    strategy_name: str = "SMA",
    strategy_kwargs: Dict[str, Any] = None,
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    position_size: float = 0.2,
    stop_loss: float = 0.05,
    take_profit: float = 0.10
) -> Dict[str, Any]:
    """Execute strategy signal generation and portfolio backtest simulation."""
    strat_kwargs = strategy_kwargs or {}
    signals_df = generate_signals(df, strategy_name=strategy_name, **strat_kwargs)

    portfolio_df, trade_log_df = run_backtest(
        signals_df,
        initial_capital=initial_capital,
        txn_cost_rate=commission_pct,
        slippage_rate=slippage_pct,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    return {
        "signals_df": signals_df,
        "portfolio_df": portfolio_df,
        "trade_log_df": trade_log_df
    }


def calculate_metrics_tool(portfolio_df: pd.DataFrame, trade_log_df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """Calculate quantitative KPI performance metrics."""
    return calculate_metrics(portfolio_df, trade_log_df, initial_capital=initial_capital)


def analyze_performance_tool(metrics: Dict[str, Any], ticker: str, strategy_name: str) -> Dict[str, Any]:
    """Analyze metrics using LLM Strategy Analyzer with structured JSON response."""
    analyzer = LLMStrategyAnalyzer()
    return analyzer.analyze_performance(metrics, ticker, strategy_name)
