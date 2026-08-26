"""
reports_page.py

Page — Quantitative Backtest & Benchmark Performance Report Generator for QUANTLAB.
Generates comprehensive report from backtester.py and benchmark.py execution results.
"""

import textwrap
import pandas as pd
import streamlit as st
from components.kpi_card import render_kpi_grid
from components.chart_container import render_equity_curve_plotly, render_drawdown_plotly


def render_reports_page(backtest_results: dict):
    st.markdown("## 📄 Quantitative Backtest & Benchmark Report")
    st.caption("Generated directly from backtester.py engine & benchmark.py comparison module")

    if not backtest_results or "metrics" not in backtest_results:
        st.info("💡 Click **🚀 Run Backtest Engine** in the sidebar to simulate strategy performance and generate a full quantitative report.")
        return

    ticker = backtest_results.get("ticker", "INFY")
    strategy = backtest_results.get("strategy_name", "SMA")
    benchmark_symbol = backtest_results.get("benchmark", "NIFTY50")
    config = backtest_results.get("config", {})
    metrics = backtest_results["metrics"]
    bench_comp = backtest_results.get("bench_comp", {})
    portfolio_df = backtest_results.get("portfolio_df")
    trade_log_df = backtest_results.get("trade_log_df")
    benchmark_df = backtest_results.get("benchmark_df")

    st.markdown("---")

    # 1. Report Header & Context
    st.markdown(f"### 📌 Report Summary: {strategy} Strategy on {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock Ticker", ticker)
    c2.metric("Strategy", strategy)
    c3.metric("Benchmark", benchmark_symbol)
    c4.metric("Initial Capital", f"₹{config.get('initial_capital', 100000.0):,.0f}")

    st.markdown("---")

    # 2. Executive Performance KPI Grid
    st.markdown("### 📊 Key Performance Indicators (KPIs)")
    render_kpi_grid(metrics, bench_comp)

    st.markdown("---")

    # 3. All Metrics Breakdown Table
    st.markdown("### 📋 Complete Quantitative Metrics Matrix")
    metrics_table = [
        {"Metric": "Total Return", "Value": f"{metrics.get('Total Return', 0.0)*100:.2f}%", "Description": "Cumulative net return over backtest period"},
        {"Metric": "CAGR (Annualized Return)", "Value": f"{metrics.get('CAGR', 0.0)*100:.2f}%", "Description": "Compound Annual Growth Rate"},
        {"Metric": "Annual Volatility", "Value": f"{metrics.get('Annual Volatility', 0.0)*100:.2f}%", "Description": "Annualized standard deviation of daily returns"},
        {"Metric": "Sharpe Ratio", "Value": f"{metrics.get('Sharpe Ratio', 0.0):.4f}", "Description": "Excess return per unit of total risk (Rf = 0.0)"},
        {"Metric": "Sortino Ratio", "Value": f"{metrics.get('Sortino Ratio', 0.0):.4f}", "Description": "Excess return per unit of downside risk"},
        {"Metric": "Calmar Ratio", "Value": f"{metrics.get('Calmar Ratio', 0.0):.4f}", "Description": "Annual return per unit of maximum drawdown"},
        {"Metric": "Maximum Drawdown", "Value": f"{metrics.get('Maximum Drawdown', 0.0)*100:.2f}%", "Description": "Worst peak-to-trough equity decline"},
        {"Metric": "Win Rate", "Value": f"{metrics.get('Win Rate', 0.0)*100:.2f}%", "Description": "Percentage of profitable round-trip trades"},
        {"Metric": "Profit Factor", "Value": f"{metrics.get('Profit Factor', 0.0):.4f}", "Description": "Gross Profit / Gross Loss ratio"},
        {"Metric": "Average Holding Period", "Value": f"{metrics.get('Average Holding Period', 0.0):.1f} Days", "Description": "Mean duration per trade"},
        {"Metric": "Average Trade Return", "Value": f"{metrics.get('Average Trade Return', 0.0)*100:.2f}%", "Description": "Mean percentage return per trade"},
        {"Metric": "Executed Trades", "Value": str(metrics.get("Number of Trades", 0)), "Description": "Total completed trade count"},
        {"Metric": "Market Exposure", "Value": f"{metrics.get('Exposure', 0.0)*100:.1f}%", "Description": "Percentage of days holding an active position"},
        {"Metric": "Daily Hit Ratio", "Value": f"{metrics.get('Hit Ratio', 0.0)*100:.1f}%", "Description": "Percentage of positive return days"}
    ]
    st.table(metrics_table)

    st.markdown("---")

    # 4. Benchmark Comparison Details
    st.markdown(f"### 🎯 Strategy vs Benchmark Comparison ({benchmark_symbol})")
    if bench_comp and bench_comp.get("status") == "SUCCESS":
        b_table = [
            {"Parameter": "Strategy CAGR", "Strategy": f"{bench_comp.get('strategy_cagr', 0)*100:.2f}%", "Benchmark": f"{bench_comp.get('benchmark_cagr', 0)*100:.2f}%"},
            {"Parameter": "Excess Return (Alpha)", "Strategy": f"{bench_comp.get('excess_return', 0)*100:+.2f}%", "Benchmark": "0.00%"},
            {"Parameter": "Beta vs Market", "Strategy": f"{bench_comp.get('beta', 1.0):.2f}", "Benchmark": "1.00"},
            {"Parameter": "Annual Volatility", "Strategy": f"{bench_comp.get('strategy_volatility', 0)*100:.2f}%", "Benchmark": f"{bench_comp.get('benchmark_volatility', 0)*100:.2f}%"},
            {"Parameter": "Sharpe Ratio", "Strategy": f"{bench_comp.get('strategy_sharpe', 0):.2f}", "Strategy Verdict": f"Outperformance: {bench_comp.get('outperformance_verdict')}"}
        ]
        st.table(b_table)
    else:
        st.info("Benchmark data comparison processed.")

    st.markdown("---")

    # 5. Visual Charts
    st.markdown("### 📈 Visual Equity & Drawdown Performance")
    if portfolio_df is not None:
        render_equity_curve_plotly(portfolio_df, benchmark_df=benchmark_df, ticker=ticker, benchmark_symbol=benchmark_symbol)
        render_drawdown_plotly(portfolio_df)

    st.markdown("---")

    # 6. Report Export / Download Button
    st.markdown("### 📥 Download Complete Project Report")

    full_report_text = textwrap.dedent(f"""
        ========================================================================
        QUANTLAB — PROJECT QUANTITATIVE BACKTEST & BENCHMARK REPORT
        ========================================================================
        Stock Ticker      : {ticker}
        Strategy          : {strategy}
        Benchmark Symbol  : {benchmark_symbol}
        Initial Capital   : INR {config.get('initial_capital', 100000.0):,.2f}
        Commission Rate   : {config.get('commission_pct', 0.001)*100:.2f}%
        Execution Slippage: {config.get('slippage_pct', 0.0005)*100:.2f}%
        
        ------------------------------------------------------------------------
        1. QUANTITATIVE PERFORMANCE METRICS
        ------------------------------------------------------------------------
        Total Return              : {metrics.get('Total Return', 0.0)*100:.2f}%
        CAGR (Annualized Return)  : {metrics.get('CAGR', 0.0)*100:.2f}%
        Annual Volatility          : {metrics.get('Annual Volatility', 0.0)*100:.2f}%
        Sharpe Ratio              : {metrics.get('Sharpe Ratio', 0.0):.4f}
        Sortino Ratio             : {metrics.get('Sortino Ratio', 0.0):.4f}
        Calmar Ratio              : {metrics.get('Calmar Ratio', 0.0):.4f}
        Maximum Drawdown          : {metrics.get('Maximum Drawdown', 0.0)*100:.2f}%
        Win Rate                  : {metrics.get('Win Rate', 0.0)*100:.2f}%
        Profit Factor             : {metrics.get('Profit Factor', 0.0):.4f}
        Average Holding Period    : {metrics.get('Average Holding Period', 0.0):.1f} Days
        Total Executed Trades     : {metrics.get('Number of Trades', 0)}
        
        ------------------------------------------------------------------------
        2. BENCHMARK COMPARISON ({benchmark_symbol})
        ------------------------------------------------------------------------
        Strategy CAGR             : {bench_comp.get('strategy_cagr', 0)*100:.2f}%
        Benchmark CAGR            : {bench_comp.get('benchmark_cagr', 0)*100:.2f}%
        Excess Return (Alpha)     : {bench_comp.get('excess_return', 0)*100:+.2f}%
        Beta vs Market            : {bench_comp.get('beta', 1.0):.2f}
        Outperformance Verdict    : {bench_comp.get('outperformance_verdict', 'N/A')}
        ========================================================================
    """)

    st.download_button(
        label="📄 Download Complete Report (.txt)",
        data=full_report_text,
        file_name=f"quantlab_report_{ticker}_{strategy}.txt",
        mime="text/plain",
        type="primary"
    )
