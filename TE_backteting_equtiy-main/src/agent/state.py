"""
state.py

LangGraph Agent State definition for Equity Backtesting Workflow.
"""

from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict
import pandas as pd


class BacktestState(TypedDict):
    experiment_id: str
    hypothesis: str
    ticker: str
    strategy_name: str
    strategy_kwargs: Dict[str, Any]
    initial_capital: float
    commission_pct: float
    slippage_pct: float
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    benchmark_symbol: str

    price_df: Optional[pd.DataFrame]
    benchmark_df: Optional[pd.DataFrame]
    signals_df: Optional[pd.DataFrame]
    portfolio_df: Optional[pd.DataFrame]
    trade_log_df: Optional[pd.DataFrame]

    metrics: Optional[Dict[str, Any]]
    benchmark_comparison: Optional[Dict[str, Any]]
    llm_analysis: Optional[Dict[str, Any]]

    results_dir: str
    report_path: str
    iteration: int
    max_iterations: int
    loop_decision: str  # "ACCEPT", "ITERATE", "STOP", "FAIL"
    experiment_history: List[Dict[str, Any]]
    final_conclusion: str
