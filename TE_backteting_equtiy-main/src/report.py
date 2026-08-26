"""
report.py

Generates institutional Strategy Fact Sheets and formatted summary reports
incorporating strategy specifications, performance, risk, trading stats, benchmark comparison,
out-of-sample validation, robustness results, and research warnings.
"""

import os
import sys
from typing import Dict, Any, List, Optional


def generate_summary_report(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_path: str,
    benchmark_comp: Optional[Dict[str, Any]] = None,
    validation_res: Optional[Dict[str, Any]] = None,
    warnings_list: Optional[List[str]] = None
) -> None:
    """
    Format and write comprehensive Strategy Fact Sheet to file.
    """
    strategy_name = config.get("strategy_name", "SMA Strategy")
    stock_ticker = config.get("stock_ticker", "INFY")
    start_date = config.get("start_date", "N/A")
    end_date = config.get("end_date", "N/A")
    initial_capital = config.get("initial_capital", 100000.0)
    final_value = config.get("final_value", 100000.0)
    hypothesis = config.get("hypothesis", "Quantitative momentum / trend following strategy")

    benchmark_comp = benchmark_comp or {}
    validation_res = validation_res or {}
    warnings_list = warnings_list or []

    # KPI values
    cagr_pct = metrics.get("CAGR", 0.0) * 100.0
    tot_ret_pct = metrics.get("Total Return", 0.0) * 100.0
    vol_pct = metrics.get("Annual Volatility", 0.0) * 100.0
    sharpe = metrics.get("Sharpe Ratio", 0.0)
    sortino = metrics.get("Sortino Ratio", 0.0)
    mdd_pct = abs(metrics.get("Maximum Drawdown", 0.0)) * 100.0
    win_rate_pct = metrics.get("Win Rate", 0.0) * 100.0
    num_trades = int(metrics.get("Number of Trades", 0))
    turnover_pct = metrics.get("Turnover", 0.0) * 100.0
    exposure_pct = metrics.get("Exposure", 0.0) * 100.0

    # Benchmark relative KPIs
    bench_symbol = benchmark_comp.get("benchmark_symbol", "NIFTY50")
    bench_cagr_pct = benchmark_comp.get("benchmark_cagr", 0.0) * 100.0
    bench_sharpe = benchmark_comp.get("benchmark_sharpe", 0.0)
    excess_cagr_pct = benchmark_comp.get("excess_return", 0.0) * 100.0
    beta = benchmark_comp.get("beta", 1.0)
    corr = benchmark_comp.get("correlation", 0.0)
    ir = benchmark_comp.get("information_ratio", 0.0)

    # Validation & Warnings
    oos_sharpe = validation_res.get("out_of_sample_metrics", {}).get("Sharpe Ratio", 0.0) if validation_res else 0.0
    sharpe_deg = validation_res.get("sharpe_degradation_pct", 0.0) if validation_res else 0.0

    formatted_warnings = "\n".join([f"  ! {w}" for w in warnings_list]) if warnings_list else "  None - All quantitative checks passed."

    fact_sheet_content = f"""================================================================================
FINAL STRATEGY FACT SHEET — SHANKH / DECUPLE INTERNSHIP 2026-27 (TE1)
================================================================================

1. STRATEGY SPECIFICATIONS & METADATA
--------------------------------------------------------------------------------
Strategy Name         : {strategy_name}
Stock Ticker / Universe: {stock_ticker}
Hypothesis            : {hypothesis}
Date Range            : {start_date} to {end_date}
Initial Capital        : INR {initial_capital:,.2f}
Final Portfolio Value  : INR {final_value:,.2f}

2. EXECUTION & PORTFOLIO RULES
--------------------------------------------------------------------------------
Position Allocation   : {config.get('position_size', 0.2)*100:.0f}% max per trade
Maximum Position Cap  : {config.get('max_positions', 10)} simultaneous positions
Rebalancing Frequency : {config.get('rebalance_freq', 'Daily')}
Transaction Cost Rate : {config.get('txn_cost_rate', 0.001)*100:.2f}% (0.1% per trade)
Execution Slippage    : {config.get('slippage_rate', 0.0005)*100:.2f}% (0.05% per trade)
Stop Loss / Take Profit: SL {config.get('stop_loss', 0.05)*100 if config.get('stop_loss') else 'None'}% | TP {config.get('take_profit', 0.10)*100 if config.get('take_profit') else 'None'}%

3. PERFORMANCE & RETURN METRICS
--------------------------------------------------------------------------------
Total Return          : {tot_ret_pct:.2f}%
CAGR (Compound Ann)   : {cagr_pct:.2f}%
Annualized Volatility : {vol_pct:.2f}%
Sharpe Ratio (Rf=0)   : {sharpe:.4f}
Sortino Ratio (Rf=0)  : {sortino:.4f}
Maximum Drawdown (MDD): -{mdd_pct:.2f}%

4. TRADING & EXPOSURE STATISTICS
--------------------------------------------------------------------------------
Total Executed Trades : {num_trades}
Trade Win Rate        : {win_rate_pct:.2f}%
Annual Portfolio Turnover: {turnover_pct:.2f}%
Market Time Exposure  : {exposure_pct:.2f}%

5. BENCHMARK COMPARISON ({bench_symbol})
--------------------------------------------------------------------------------
Strategy CAGR vs Benchmark: {cagr_pct:.2f}% vs {bench_cagr_pct:.2f}%
Excess Annual Return   : {excess_cagr_pct:+.2f}%
Strategy Sharpe vs Bench  : {sharpe:.2f} vs {bench_sharpe:.2f}
Portfolio Beta        : {beta:.2f}
Correlation with Bench: {corr:.2f}
Information Ratio     : {ir:.2f}

6. OUT-OF-SAMPLE VALIDATION & ROBUSTNESS
--------------------------------------------------------------------------------
Out-of-Sample Sharpe  : {oos_sharpe:.4f}
Sharpe Degradation %  : {sharpe_deg:.1f}%
Validation Status     : {"PASSED" if validation_res.get("passed_validation", True) else "FAILED"}

7. RESEARCH VALIDATION WARNINGS & RISK AUDIT
--------------------------------------------------------------------------------
{formatted_warnings}

================================================================================
Generated automatically by Equity Backtesting Framework.
================================================================================
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(fact_sheet_content)
