"""
results_page.py

Page 7 — Detailed Backtest Results Terminal for QUANTLAB.
"""

import streamlit as st
from components.kpi_card import render_kpi_grid
from components.chart_container import (
    render_equity_curve_plotly,
    render_drawdown_plotly,
    render_monthly_returns_plotly
)
from llm.analyzer import LLMStrategyAnalyzer


def render_results_page(backtest_results: dict):
    if not backtest_results or "metrics" not in backtest_results:
        st.info("💡 Please run a backtest simulation first from the Backtest Config page.")
        return

    ticker = backtest_results.get("ticker", "INFY")
    strategy = backtest_results.get("strategy_name", "SMA")
    metrics = backtest_results["metrics"]
    bench_comp = backtest_results.get("bench_comp", {})
    portfolio_df = backtest_results.get("portfolio_df")
    trade_log_df = backtest_results.get("trade_log_df")
    benchmark_df = backtest_results.get("benchmark_df")

    st.markdown(f"## 📈 Backtest Results: {strategy} Strategy on {ticker}")
    st.caption("Quantitative Performance Breakdown & Risk Metrics")

    # KPI Summary Row
    render_kpi_grid(metrics, bench_comp)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Performance", "Trades", "Risk", "AI Analysis"])

    with tab1:
        st.markdown("### Strategy vs Benchmark Equity Growth")
        if portfolio_df is not None:
            render_equity_curve_plotly(
                portfolio_df,
                benchmark_df=benchmark_df,
                ticker=ticker,
                benchmark_symbol=backtest_results.get("benchmark", "NIFTY50")
            )

    with tab2:
        st.markdown("### Peak-to-Trough Drawdown Curve")
        if portfolio_df is not None:
            render_drawdown_plotly(portfolio_df)

        st.markdown("### Monthly Return Heatmap")
        if portfolio_df is not None:
            render_monthly_returns_plotly(portfolio_df)

    with tab3:
        st.markdown("### Executed Trade Log")
        if trade_log_df is not None and not trade_log_df.empty:
            st.dataframe(trade_log_df, use_container_width=True)
        else:
            st.info("No trades executed during this backtest period.")

    with tab4:
        st.markdown("### Detailed Risk Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Volatility", f"{metrics.get('Annual Volatility', 0)*100:.2f}%")
        c2.metric("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.2f}")
        c3.metric("Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.2f}")

    with tab5:
        st.markdown("### AI Strategy Analysis")
        analyzer = LLMStrategyAnalyzer()
        res = analyzer.analyze_performance(metrics, ticker, strategy)

        st.markdown("#### Rating & Verdict")
        st.write(f"**Rating:** {res.get('rating')} | **Risk Profile:** {res.get('risk')}")
        st.markdown(res.get("executive_summary", ""))

        with st.expander("Full Institutional Analysis JSON", expanded=False):
            st.json(res)
