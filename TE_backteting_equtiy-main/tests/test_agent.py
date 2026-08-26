"""
test_agent.py

Unit test suite for LangGraph Agent StateGraph and Loop Engineering state machine.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent.workflow import run_agent_workflow
from agent.loop import evaluate_research_iteration, execute_controlled_iteration
from agent.state import BacktestState


def test_agent_workflow_execution():
    """Test full agent workflow execution."""
    state = run_agent_workflow(
        ticker="INFY",
        strategy_name="SMA",
        max_iterations=2,
        initial_capital=100000.0
    )

    assert state["ticker"] == "INFY"
    assert state["iteration"] >= 1
    assert state["loop_decision"] in ["ACCEPT", "ITERATE", "STOP"]
    assert state["metrics"] is not None
    assert "Sharpe Ratio" in state["metrics"]


def test_loop_engineering_iteration_ceiling():
    """Test that Loop Engineering respects max_iterations stopping condition."""
    dummy_state: BacktestState = {
        "experiment_id": "TEST_EXP",
        "hypothesis": "Test hypothesis",
        "ticker": "INFY",
        "strategy_name": "SMA",
        "strategy_kwargs": {"short_window": 20, "long_window": 50},
        "initial_capital": 100000.0,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "position_size": 0.2,
        "stop_loss": 0.05,
        "take_profit": 0.10,
        "benchmark_symbol": "NIFTY50",
        "price_df": None,
        "benchmark_df": None,
        "signals_df": None,
        "portfolio_df": None,
        "trade_log_df": None,
        "metrics": {"Sharpe Ratio": 0.40, "CAGR": 0.05, "Maximum Drawdown": -0.25},
        "benchmark_comparison": None,
        "llm_analysis": None,
        "results_dir": "results",
        "report_path": "",
        "iteration": 3,
        "max_iterations": 3,
        "loop_decision": "PENDING",
        "experiment_history": [],
        "final_conclusion": ""
    }

    evaluated_state = evaluate_research_iteration(dummy_state)
    assert evaluated_state["loop_decision"] == "STOP"
