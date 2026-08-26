"""
benchmark_page.py

Page 7 — Benchmark Analysis Page Renderer.
Compares strategy return and risk metrics against Nifty 50 index with clear conclusion.
"""

import textwrap
import pandas as pd
import streamlit as st
from components.chart_container import render_equity_curve_plotly

def render_benchmark_page(backtest_results: dict):
    """
    Render Benchmark Comparison Page.
    """
    st.markdown("## 🎯 Nifty 50 Benchmark Relative Analysis")
    st.markdown("<div style='color: #94a3b8; font-size: 1.05rem; margin-bottom: 20px;'>Evaluates alpha generation, Beta market risk, correlation, and tracking error against Nifty 50.</div>", unsafe_allow_html=True)

    if not backtest_results or "bench_comp" not in backtest_results:
        st.info("Please run a backtest to generate benchmark comparison data.")
        return

    bench_comp = backtest_results["bench_comp"]
    portfolio_df = backtest_results.get("portfolio_df")
    benchmark_df = backtest_results.get("benchmark_df")
    ticker = backtest_results.get("ticker", "INFY")
    benchmark_symbol = backtest_results.get("benchmark", "NIFTY50")

    if bench_comp.get("status") == "ERROR":
        st.warning(f"Benchmark comparison error: {bench_comp.get('warning')}")
        return

    excess_cagr = bench_comp.get("excess_return", 0.0) * 100.0
    outperformed = excess_cagr > 0.0

    if outperformed:
        conclusion_html = textwrap.dedent(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 1rem 1.5rem; border-radius: 0.75rem; margin-bottom: 20px;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #10b981;">
                    🎉 OUTPERFORMANCE VERIFIED
                </div>
                <div style="color: #f8fafc; font-size: 1.0rem; margin-top: 4px;">
                    Strategy <b>outperformed {benchmark_symbol}</b> by <b style="color: #10b981;">{excess_cagr:+.2f}% CAGR</b> over the backtest period.
                </div>
            </div>
        """)
    else:
        conclusion_html = textwrap.dedent(f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); padding: 1rem 1.5rem; border-radius: 0.75rem; margin-bottom: 20px;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #ef4444;">
                    ⚠️ UNDERPERFORMANCE DETECTED
                </div>
                <div style="color: #f8fafc; font-size: 1.0rem; margin-top: 4px;">
                    Strategy <b>underperformed {benchmark_symbol}</b> by <b style="color: #ef4444;">{excess_cagr:.2f}% CAGR</b>.
                </div>
            </div>
        """)
    st.markdown(conclusion_html, unsafe_allow_html=True)

    col_metrics, col_chart = st.columns([1, 1])

    with col_metrics:
        st.markdown("### 📊 Relative Metrics Breakdown")
        
        comp_df = pd.DataFrame({
            "Metric": ["CAGR (%)", "Sharpe Ratio", "Annual Volatility (%)", "Max Drawdown (%)"],
            f"Strategy ({ticker})": [
                f"{bench_comp.get('strategy_cagr', 0.0)*100:.2f}%",
                f"{bench_comp.get('strategy_sharpe', 0.0):.2f}",
                f"{bench_comp.get('strategy_volatility', 0.0)*100:.2f}%",
                f"{bench_comp.get('strategy_max_drawdown', 0.0)*100:.2f}%"
            ],
            f"Benchmark ({benchmark_symbol})": [
                f"{bench_comp.get('benchmark_cagr', 0.0)*100:.2f}%",
                f"{bench_comp.get('benchmark_sharpe', 0.0):.2f}",
                f"{bench_comp.get('benchmark_volatility', 0.0)*100:.2f}%",
                f"{bench_comp.get('benchmark_max_drawdown', 0.0)*100:.2f}%"
            ]
        })
        st.dataframe(comp_df, use_container_width=True)

        details_html = textwrap.dedent(f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 0.5rem; font-size: 0.9rem; line-height: 1.6; border: 1px solid rgba(255,255,255,0.08);">
                <div>⚡ <b>Beta vs Benchmark:</b> {bench_comp.get('beta', 1.0):.2f}</div>
                <div>🔗 <b>Correlation:</b> {bench_comp.get('correlation', 0.0):.2f}</div>
                <div>📐 <b>Tracking Error:</b> {bench_comp.get('tracking_error', 0.0)*100:.2f}%</div>
                <div>🏆 <b>Information Ratio:</b> {bench_comp.get('information_ratio', 0.0):.2f}</div>
            </div>
        """)
        st.markdown(details_html, unsafe_allow_html=True)

    with col_chart:
        if portfolio_df is not None:
            render_equity_curve_plotly(portfolio_df, benchmark_df, ticker, benchmark_symbol)
