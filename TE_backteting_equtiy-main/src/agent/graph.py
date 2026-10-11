"""
graph.py

LangGraph StateGraph builder for Equity Strategy Research Workflow.
"""

import os
import sys
import pandas as pd
from typing import Optional, cast, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent.state import BacktestState
from agent.tools import run_data_loader_tool, run_backtest_tool, calculate_metrics_tool, analyze_performance_tool
from agent.loop import evaluate_research_iteration, execute_controlled_iteration, route_loop_decision
from visualization import plot_equity_curve, plot_drawdown

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


def node_load_data(state: BacktestState) -> BacktestState:
    """Node 1: Data Loader Tool call."""
    ticker = state["ticker"]
    df = run_data_loader_tool(ticker)
    state["price_df"] = df
    print(f"[Graph Node: Data Loader] Loaded {len(df)} price rows for {ticker}")
    return state


def node_run_backtest(state: BacktestState) -> BacktestState:
    """Node 2: Backtest Tool call."""
    price_df = state.get("price_df")
    if price_df is None:
        raise ValueError("price_df is uninitialized in BacktestState")
    res = run_backtest_tool(
        price_df,
        strategy_name=state["strategy_name"],
        strategy_kwargs=state.get("strategy_kwargs"),
        initial_capital=state.get("initial_capital", 100000.0),
        commission_pct=state.get("commission_pct", 0.001),
        slippage_pct=state.get("slippage_pct", 0.0005),
        position_size=state.get("position_size", 0.2),
        stop_loss=state.get("stop_loss", 0.05),
        take_profit=state.get("take_profit", 0.10)
    )
    state["signals_df"] = res["signals_df"]
    state["portfolio_df"] = res["portfolio_df"]
    state["trade_log_df"] = res["trade_log_df"]
    print(f"[Graph Node: Backtest] Iteration {state.get('iteration', 1)} executed. Trades: {len(res['trade_log_df'])}")
    return state


def node_calculate_metrics(state: BacktestState) -> BacktestState:
    """Node 3: Metrics Tool call."""
    portfolio_df = state.get("portfolio_df")
    trade_log_df = state.get("trade_log_df")
    if portfolio_df is None or trade_log_df is None:
        raise ValueError("portfolio_df or trade_log_df is uninitialized in BacktestState")
    metrics = calculate_metrics_tool(
        portfolio_df,
        trade_log_df,
        initial_capital=state.get("initial_capital", 100000.0)
    )
    state["metrics"] = metrics
    print(f"[Graph Node: Metrics] Sharpe: {metrics['Sharpe Ratio']:.2f}, CAGR: {metrics['CAGR']*100:.2f}%")
    return state


def node_evaluator(state: BacktestState) -> BacktestState:
    """Node 4: Loop Engineering Evaluator Node."""
    return evaluate_research_iteration(state)


def node_controlled_iteration(state: BacktestState) -> BacktestState:
    """Node 5: Controlled Parameter Iteration Node."""
    return execute_controlled_iteration(state)


def node_generate_artifacts(state: BacktestState) -> BacktestState:
    """Node 6: LLM Analysis & Visualization Artifact Generator."""
    ticker = state["ticker"]
    metrics = state.get("metrics")
    portfolio_df = state.get("portfolio_df")

    if metrics is None:
        raise ValueError("metrics is uninitialized in BacktestState")
    if portfolio_df is None:
        raise ValueError("portfolio_df is uninitialized in BacktestState")

    # LLM Strategy Analysis
    llm_analysis = analyze_performance_tool(metrics, ticker, state["strategy_name"])
    state["llm_analysis"] = llm_analysis

    res_dir = state.get("results_dir", os.path.join(BASE_DIR, "results"))
    os.makedirs(res_dir, exist_ok=True)

    try:
        plot_equity_curve(portfolio_df, ticker, os.path.join(res_dir, "equity_curve.png"))
        plot_drawdown(portfolio_df, os.path.join(res_dir, "drawdown.png"))
    except Exception as e:
        print(f"[Graph Node: Artifacts] Chart warning: {e}")

    state["final_conclusion"] = f"Workflow completed after {state.get('iteration', 1)} iteration(s). Verdict: {llm_analysis.get('rating', 'Completed')}"
    print(f"[Graph Node: Artifacts] Final analysis generated successfully.")
    return state


def build_agent_graph():
    """Build and compile the LangGraph StateGraph."""
    if not HAS_LANGGRAPH:
        return None

    builder = StateGraph(cast(Any, BacktestState))

    builder.add_node("load_data", node_load_data)
    builder.add_node("run_backtest", node_run_backtest)
    builder.add_node("calculate_metrics", node_calculate_metrics)
    builder.add_node("research_evaluator", node_evaluator)
    builder.add_node("controlled_iteration", node_controlled_iteration)
    builder.add_node("generate_artifacts", node_generate_artifacts)

    builder.set_entry_point("load_data")

    builder.add_edge("load_data", "run_backtest")
    builder.add_edge("run_backtest", "calculate_metrics")
    builder.add_edge("calculate_metrics", "research_evaluator")

    builder.add_conditional_edges(
        "research_evaluator",
        route_loop_decision,
        {
            "iterate": "controlled_iteration",
            "finalize": "generate_artifacts"
        }
    )

    builder.add_edge("controlled_iteration", "run_backtest")
    builder.add_edge("generate_artifacts", END)

    return builder.compile()
