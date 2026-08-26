"""
main.py

Main orchestrator CLI for the Equity Backtesting System.
Executes the agent-driven backtest pipeline:
1. Load data
2. Generate strategy signals (SMA / EMA / RSI)
3. Run simulation with costs, position sizing, SL, TP
4. Calculate quantitative metrics
5. Generate charts (Signals, Equity Curve, Drawdown, Rolling Sharpe, Monthly Returns, Trade Dist, Allocation)
6. Call LLM Strategy Analyzer
7. Generate final summary report

Usage:
    python src/main.py RELIANCE --strategy SMA --position-size 0.2 --stop-loss 0.05 --take-profit 0.10
"""

import argparse
import os
import sys
from typing import Dict, Any

# Ensure local path imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agent.workflow import run_agent_workflow


def run_pipeline(
    ticker: str = "INFY",
    strategy_name: str = "SMA",
    initial_capital: float = 100000.0,
    txn_cost_rate: float = 0.001,
    position_size: float = 0.2,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    **kwargs
) -> Dict[str, Any]:
    """
    Run full end-to-end backtest pipeline.
    Maintains backward compatibility while executing the node-based workflow.
    """
    print("=" * 80)
    print(f"STARTING BACKTEST PIPELINE FOR: {ticker} ({strategy_name})")
    print("=" * 80)

    final_state = run_agent_workflow(
        ticker=ticker,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        commission_pct=txn_cost_rate,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        **kwargs
    )

    print("=" * 80)
    print("BACKTEST PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)
    return final_state["metrics"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Equity Backtesting Orchestrator.")
    parser.add_argument("ticker", type=str, nargs="?", default="INFY", help="Stock ticker (default: INFY)")
    parser.add_argument("--strategy", type=str, default="SMA", choices=["SMA", "EMA", "RSI"], help="Strategy name")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (default: 0.001)")
    parser.add_argument("--position-size", type=float, default=0.2, help="Position size fraction (default: 0.2)")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss fraction (default: 0.05)")
    parser.add_argument("--take-profit", type=float, default=0.10, help="Take profit fraction (default: 0.10)")

    args = parser.parse_args()

    try:
        run_pipeline(
            ticker=args.ticker,
            strategy_name=args.strategy,
            initial_capital=args.capital,
            txn_cost_rate=args.commission,
            position_size=args.position_size,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        sys.exit(1)
