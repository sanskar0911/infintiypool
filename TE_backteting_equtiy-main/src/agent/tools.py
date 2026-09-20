"""
tools.py

LangGraph Executable Tools for Backtesting, Execution Analysis, Performance Metrics & LLM Analysis.
Exposes deterministic engine helper tools for AI Agent workflow integration.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, Optional

# Ensure path imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import load_stock_data
from strategy import generate_signals
from engine import BacktestEngine, BacktestConfig, ExecutionTiming, InsufficientCashPolicy
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
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    position_size: float = 0.2,
    stop_loss: Optional[float] = 0.05,
    take_profit: Optional[float] = 0.10,
    execution_timing: str = "NEXT_BAR_OPEN"
) -> Dict[str, Any]:
    """Execute strategy signal generation and realistic engine backtest simulation."""
    strat_kwargs = strategy_kwargs or {}
    signals_df = generate_signals(df, strategy_name=strategy_name, **strat_kwargs)

    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        commission_bps=commission_pct * 10000.0,
        slippage_bps=slippage_pct * 10000.0,
        execution_timing=ExecutionTiming(execution_timing) if isinstance(execution_timing, str) else execution_timing,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    engine = BacktestEngine(config)
    res = engine.run(signals_df)

    return {
        "signals_df": signals_df,
        "portfolio_df": res.portfolio_history,
        "trade_log_df": res.trade_log_df,
        "execution_statistics": res.execution_statistics,
        "result_object": res
    }


def calculate_metrics_tool(portfolio_df: pd.DataFrame, trade_log_df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """Calculate quantitative KPI performance metrics."""
    return calculate_metrics(portfolio_df, trade_log_df, initial_capital=initial_capital)


def get_execution_statistics_tool(res: Any) -> Dict[str, Any]:
    """Retrieve execution statistics from BacktestResult object."""
    if hasattr(res, "get_execution_statistics"):
        return res.get_execution_statistics()
    return {}


def get_portfolio_state_tool(res: Any) -> Dict[str, Any]:
    """Retrieve final portfolio state snapshot from BacktestResult object."""
    if hasattr(res, "get_portfolio_state"):
        return res.get_portfolio_state()
    return {}


def analyze_performance_tool(metrics: Dict[str, Any], ticker: str, strategy_name: str) -> Dict[str, Any]:
    """Analyze metrics using LLM Strategy Analyzer with structured JSON response."""
    analyzer = LLMStrategyAnalyzer()
    return analyzer.analyze_performance(metrics, ticker, strategy_name)
