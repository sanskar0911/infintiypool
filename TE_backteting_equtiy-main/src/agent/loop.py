"""
loop.py

Loop Engineering decision state machine for LangGraph research agent.

Workflow:
                 USER / HYPOTHESIS
                         ↓
                     BASELINE
                         ↓
                     TOOL CALL
                         ↓
                    ANALYSIS
                         ↓
                  EVALUATION
                         ↓
                  ┌──────┴──────┐
                  │             │
                PASS          ITERATE
                  │             │
                  ↓             ↓
                FINAL      CONTROLLED
                RESULT     RE-EVALUATION
                                │
                                ↓
                              STOP
"""

from typing import Dict, Any
from agent.state import BacktestState


def evaluate_research_iteration(state: BacktestState) -> BacktestState:
    """
    Research Evaluator Node:
    Evaluates current iteration metrics against target hypothesis criteria.
    Determines loop decision: ACCEPT, ITERATE, or STOP.
    """
    curr_iter = state.get("iteration", 1)
    max_iter = state.get("max_iterations", 3)
    metrics = state.get("metrics") or {}

    sharpe = metrics.get("Sharpe Ratio", 0.0)
    cagr = metrics.get("CAGR", 0.0)
    max_dd = abs(metrics.get("Maximum Drawdown", 0.0))

    # Log current iteration entry into history
    history_entry = {
        "iteration": curr_iter,
        "strategy": state["strategy_name"],
        "params": dict(state.get("strategy_kwargs", {})),
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd
    }
    history = list(state.get("experiment_history") or [])
    history.append(history_entry)
    state["experiment_history"] = history

    # Loop Decision State Machine Rules
    if sharpe >= 0.8 and cagr >= 0.10 and max_dd <= 0.20:
        state["loop_decision"] = "ACCEPT"
        print(f"[Loop Evaluator] Iteration {curr_iter}: ACCEPT — Sharpe {sharpe:.2f} satisfies criteria.")
    elif curr_iter < max_iter:
        state["loop_decision"] = "ITERATE"
        print(f"[Loop Evaluator] Iteration {curr_iter}: ITERATE — Sharpe {sharpe:.2f} below target 0.80. Triggering controlled re-evaluation.")
    else:
        state["loop_decision"] = "STOP"
        print(f"[Loop Evaluator] Iteration {curr_iter}: STOP — Reached maximum iterations limit ({max_iter}).")

    return state


def execute_controlled_iteration(state: BacktestState) -> BacktestState:
    """
    Controlled Parameter Iteration Node:
    Adjusts strategy parameters deterministically for the next iteration.
    Enforces maximum iteration ceiling.
    """
    curr_iter = state.get("iteration", 1) + 1
    state["iteration"] = curr_iter

    strat_name = state["strategy_name"]
    strat_kwargs = dict(state.get("strategy_kwargs", {}))

    # Controlled Parameter Variations (No black-box or unbounded changes)
    if strat_name == "SMA":
        strat_kwargs["short_window"] = strat_kwargs.get("short_window", 20) + 5
        strat_kwargs["long_window"] = strat_kwargs.get("long_window", 50) + 10
    elif strat_name == "EMA":
        strat_kwargs["short_window"] = strat_kwargs.get("short_window", 12) + 4
        strat_kwargs["long_window"] = strat_kwargs.get("long_window", 26) + 8
    elif strat_name == "RSI":
        strat_kwargs["overbought"] = max(65, strat_kwargs.get("overbought", 70) - 2)
        strat_kwargs["oversold"] = min(35, strat_kwargs.get("oversold", 30) + 2)

    state["strategy_kwargs"] = strat_kwargs
    print(f"[Controlled Iteration] Iteration {curr_iter} parameters updated: {strat_kwargs}")
    return state


def route_loop_decision(state: BacktestState) -> str:
    """Conditional Edge router based on loop_decision."""
    decision = state.get("loop_decision", "STOP")
    if decision == "ITERATE":
        return "iterate"
    return "finalize"
