"""
demo.py

Reproducible Demo Scenario for QUANTLAB Upgraded Backtesting Engine.
Runs an end-to-end backtest on historical data comparing:
WITHOUT COSTS (Ideal Zero Friction) vs WITH COSTS + SLIPPAGE (Realistic Execution).
"""

import os
import sys
import pandas as pd

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from strategy import generate_signals
from engine import BacktestEngine, BacktestConfig, ExecutionTiming


def run_demo():
    print("=" * 80)
    print("QUANTLAB — UPGRADED BACKTESTING ENGINE DEMO SCENARIO")
    print("Pipeline: SIGNAL -> POSITION SIZING -> ORDER -> EXECUTION -> FILL -> PORTFOLIO -> METRICS")
    print("=" * 80)

    # Load data
    data_path = os.path.join(BASE_DIR, "data", "processed", "RELIANCE.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, "data", "processed", "INFY.csv")

    print(f"\n1. Loading Market Data: {os.path.basename(data_path)}")
    df = pd.read_csv(data_path, index_col="Date")
    df.index = pd.to_datetime(df.index)

    # Generate signals
    print("2. Generating Strategy Signals (SMA Crossover 20/50)...")
    signals_df = generate_signals(df, strategy_name="SMA", short_window=20, long_window=50)

    # Configuration 1: Realistic Execution
    realistic_config = BacktestConfig(
        initial_capital=100000.0,
        position_size=0.20,
        slippage_bps=10.0,       # 10 bps = 0.10%
        commission_bps=5.0,      # 5 bps = 0.05%
        execution_timing=ExecutionTiming.NEXT_BAR_OPEN
    )

    print("\n3. Executing REALISTIC Backtest Engine...")
    print(f"   Initial Capital: INR {realistic_config.initial_capital:,.2f}")
    print(f"   Position Size:   {realistic_config.position_size*100:.0f}%")
    print(f"   Execution Mode:  {realistic_config.execution_timing}")
    print(f"   Slippage:        {realistic_config.slippage_bps} bps (0.10%)")
    print(f"   Commission:      {realistic_config.commission_bps} bps (0.05%)")

    engine_realistic = BacktestEngine(realistic_config)
    res_realistic = engine_realistic.run(signals_df, symbol="RELIANCE")

    # Configuration 2: Zero Costs Execution
    ideal_config = BacktestConfig(
        initial_capital=100000.0,
        position_size=0.20,
        slippage_bps=0.0,
        commission_bps=0.0,
        execution_timing=ExecutionTiming.NEXT_BAR_OPEN
    )

    print("\n4. Executing ZERO-COST (Ideal) Backtest Engine...")
    engine_ideal = BacktestEngine(ideal_config)
    res_ideal = engine_ideal.run(signals_df, symbol="RELIANCE")

    # Execution Trace Sample
    print("\n5. Sample Execution Trace Log (First 6 Execution Steps):")
    print("-" * 80)
    for trace_item in res_realistic.execution_trace[:6]:
        print(f"  * {trace_item.replace('₹', 'INR ')}")
    print("-" * 80)

    # Comparative Execution Impact Table
    m_real = res_realistic.metrics
    m_ideal = res_ideal.metrics

    print("\n6. EXECUTION IMPACT ANALYSIS: REALISTIC VS WITHOUT COSTS")
    print("=" * 80)
    print(f"{'Metric':<25} | {'WITHOUT COSTS':<18} | {'REALISTIC (10bps/5bps)':<22} | {'Diff / Impact'}")
    print("-" * 80)
    
    comp_metrics = ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Win Rate"]
    for k in comp_metrics:
        val_ideal = m_ideal.get(k, 0.0)
        val_real = m_real.get(k, 0.0)
        if "Return" in k or "CAGR" in k or "Drawdown" in k or "Rate" in k:
            str_ideal = f"{val_ideal*100:.2f}%"
            str_real = f"{val_real*100:.2f}%"
            diff = f"{(val_real - val_ideal)*100:+.2f}%"
        else:
            str_ideal = f"{val_ideal:.2f}"
            str_real = f"{val_real:.2f}"
            diff = f"{(val_real - val_ideal):+.2f}"

        print(f"{k:<25} | {str_ideal:<18} | {str_real:<22} | {diff}")

    print("=" * 80)
    print(f"Total Fills Executed: {res_realistic.execution_statistics.get('Total Fills', 0)}")
    print(f"Total Commission Paid: INR {res_realistic.execution_statistics.get('Total Transaction Costs', 0.0):,.2f}")
    print(f"Total Slippage Cost:  INR {res_realistic.execution_statistics.get('Total Slippage Cost', 0.0):,.2f}")
    print("=" * 80)
    print("Demo completed successfully!\n")


if __name__ == "__main__":
    run_demo()
