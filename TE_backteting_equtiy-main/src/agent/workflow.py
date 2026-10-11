"""
workflow.py

Master Entry Point for LangGraph Agent Workflow with Loop Engineering.
"""

import os
import sys
from typing import Dict, Any, Optional, cast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent.state import BacktestState
from agent.graph import (
    build_agent_graph,
    node_load_data,
    node_run_backtest,
    node_calculate_metrics,
    node_evaluator,
    node_controlled_iteration,
    node_generate_artifacts
)
from agent.loop import route_loop_decision


def run_agent_workflow(
    ticker: str = "INFY",
    strategy_name: str = "SMA",
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    position_size: float = 0.2,
    stop_loss: Optional[float] = 0.05,
    take_profit: Optional[float] = 0.10,
    benchmark_symbol: str = "NIFTY50",
    max_iterations: int = 3,
    hypothesis: str = "Test quantitative momentum edge on Indian equities",
    **strategy_kwargs
) -> BacktestState:
    """Execute LangGraph agent research workflow with Loop Engineering."""
    initial_state: BacktestState = {
        "experiment_id": f"EXP_{ticker}_{strategy_name}",
        "hypothesis": hypothesis,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "strategy_kwargs": strategy_kwargs or ({"short_window": 20, "long_window": 50} if strategy_name == "SMA" else {}),
        "initial_capital": initial_capital,
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct,
        "position_size": position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "benchmark_symbol": benchmark_symbol,
        "price_df": None,
        "benchmark_df": None,
        "signals_df": None,
        "portfolio_df": None,
        "trade_log_df": None,
        "metrics": None,
        "benchmark_comparison": None,
        "llm_analysis": None,
        "results_dir": os.path.join(BASE_DIR, "results"),
        "report_path": "",
        "iteration": 1,
        "max_iterations": max_iterations,
        "loop_decision": "PENDING",
        "experiment_history": [],
        "final_conclusion": ""
    }

    graph = build_agent_graph()
    if graph is not None:
        print("[Workflow Runner] Executing via compiled LangGraph StateGraph...")
        return cast(BacktestState, graph.invoke(initial_state))
    else:
        print("[Workflow Runner] Executing via Sequential Loop State Machine Fallback...")
        s = node_load_data(initial_state)
        while True:
            s = node_run_backtest(s)
            s = node_calculate_metrics(s)
            s = node_evaluator(s)
            route = route_loop_decision(s)
            if route == "iterate":
                s = node_controlled_iteration(s)
            else:
                break
        s = node_generate_artifacts(s)
        return s


if __name__ == "__main__":
    print("Testing LangGraph Agent Workflow with Loop Engineering...")
    state = run_agent_workflow("RELIANCE", "SMA", max_iterations=2)
    print("\nWorkflow Execution Complete!")
    print(f"Total Iterations: {state['iteration']}")
    print(f"Final Decision: {state['loop_decision']}")
    if state.get("llm_analysis"):
        print(f"LLM Rating: {state['llm_analysis'].get('rating')}")
