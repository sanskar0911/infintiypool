"""
results_page.py

Page 7 — Detailed Backtest Results & Execution Pipeline Visualizer for QUANTLAB.
Renders Execution Pipeline Cards, Portfolio Breakdown, Execution Statistics,
Impact Analysis (Realistic vs Zero-Cost), and Debug Trace Log.
"""

import streamlit as st
import pandas as pd
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
    exec_stats = backtest_results.get("execution_statistics", {})
    exec_trace = backtest_results.get("execution_trace", [])

    st.markdown(f"## 📈 Backtest Results: {strategy} Strategy on {ticker}")
    st.caption("Quantitative Performance Breakdown, Execution Layer Statistics & Risk Metrics")

    # KPI Summary Row
    render_kpi_grid(metrics, bench_comp)

    st.markdown("---")

    # 1. EXECUTION PIPELINE VISUALIZATION (Section 22)
    st.markdown("### 🔄 EXECUTION PIPELINE ARCHITECTURE")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.info("**1. SIGNAL**\n\nBUY / SELL / HOLD\nGenerated at Bar T")
    with p_col2:
        st.info("**2. ORDER**\n\nMARKET Order\nPosition Sized Qty")
    with p_col3:
        st.info("**3. EXECUTION**\n\nNext Bar Open\nPoint-in-Time Safe")
    with p_col4:
        st.info("**4. FILL**\n\nSlippage + Fees\nCanonical Trade Record")
    with p_col5:
        st.info("**5. PORTFOLIO**\n\nCash + Positions\nRealized/Unrealized P&L")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", "Performance", "Trades & Details", "Portfolio State", "Execution Stats", "Impact Analysis", "AI Analysis"
    ])

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
        st.markdown("### Executed Trade Log & Details Inspector")
        if trade_log_df is not None and not trade_log_df.empty:
            st.dataframe(trade_log_df, use_container_width=True)
            
            # Detailed Trade Inspector (Section 24)
            st.markdown("#### 🔍 Single Trade Inspector")
            trade_idx = st.number_input("Select Trade Index", min_value=0, max_value=max(0, len(trade_log_df)-1), value=0, step=1)
            if len(trade_log_df) > 0:
                trade_row = trade_log_df.iloc[trade_idx]
                t_cols1, t_cols2, t_cols3 = st.columns(3)
                with t_cols1:
                    st.write(f"**Date:** {trade_row.name}")
                    st.write(f"**Symbol:** {ticker}")
                    st.write(f"**Side:** {trade_row.get('BUY/SELL', 'N/A')}")
                with t_cols2:
                    st.write(f"**Reference Price:** ₹{trade_row.get('Price', 0.0):,.2f}")
                    st.write(f"**Fill Price:** ₹{trade_row.get('Execution Price', 0.0):,.2f}")
                    st.write(f"**Shares:** {trade_row.get('Shares', 0)}")
                with t_cols3:
                    st.write(f"**Slippage Cost:** ₹{trade_row.get('Slippage', 0.0):,.2f}")
                    st.write(f"**Transaction Fee:** ₹{trade_row.get('Commission', 0.0):,.2f}")
                    st.write(f"**Total Trade Value:** ₹{trade_row.get('Value', 0.0):,.2f}")
        else:
            st.info("No trades executed during this backtest period.")

    with tab4:
        st.markdown("### Portfolio State & Valuation Breakdown")
        if portfolio_df is not None and not portfolio_df.empty:
            last_p = portfolio_df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Cash Balance", f"₹{last_p.get('Cash', 0.0):,.2f}")
            c2.metric("Positions Value", f"₹{last_p.get('Holdings Value', 0.0):,.2f}")
            c3.metric("Total Portfolio Value", f"₹{last_p.get('Portfolio Value', 0.0):,.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Realized P&L", f"₹{last_p.get('Realized PnL', 0.0):,.2f}")
            c5.metric("Unrealized P&L", f"₹{last_p.get('Unrealized PnL', 0.0):,.2f}")
            c6.metric("Exposure", f"{last_p.get('Exposure', 0.0)*100:.2f}%")

            st.markdown("#### Portfolio Valuation Time-Series")
            st.line_chart(portfolio_df[["Cash", "Holdings Value", "Portfolio Value"]])

    with tab5:
        st.markdown("### Execution Statistics")
        if exec_stats:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total Orders", f"{exec_stats.get('Total Orders', 0)}")
            e2.metric("Total Fills", f"{exec_stats.get('Total Fills', 0)}")
            e3.metric("Buy Fills", f"{exec_stats.get('Buy Fills', 0)}")
            e4.metric("Sell Fills", f"{exec_stats.get('Sell Fills', 0)}")

            e5, e6, e7, e8 = st.columns(4)
            e5.metric("Total Commission Fees", f"₹{exec_stats.get('Total Transaction Costs', 0.0):,.2f}")
            e6.metric("Total Slippage Cost", f"₹{exec_stats.get('Total Slippage Cost', 0.0):,.2f}")
            e7.metric("Average Slippage", f"₹{exec_stats.get('Average Slippage', 0.0):,.2f}")
            e8.metric("Turnover", f"{exec_stats.get('Turnover', 0.0):.2f}x")

        with st.expander("🐛 Debug Execution Trace Log", expanded=False):
            if exec_trace:
                st.code("\n".join(exec_trace), language="text")
            else:
                st.caption("No trace logs recorded.")

    with tab6:
        st.markdown("### Execution Impact Analysis (WITHOUT COSTS vs WITH COSTS)")
        st.caption("Demonstrates performance decay from realistic slippage and brokerage fees")
        
        # Calculate impact summary
        tot_fee = exec_stats.get("Total Transaction Costs", 0.0) + exec_stats.get("Total Slippage Cost", 0.0)
        cagr_val = metrics.get("CAGR", 0.0)
        sharpe_val = metrics.get("Sharpe Ratio", 0.0)

        st.markdown(f"""
        - **Total Friction Loss:** ₹{tot_fee:,.2f}
        - **Net CAGR (With Friction):** {cagr_val*100:.2f}%
        - **Net Sharpe Ratio:** {sharpe_val:.2f}
        """)

    with tab7:
        st.markdown("### AI Strategy Analysis")
        analyzer = LLMStrategyAnalyzer()
        res = analyzer.analyze_performance(metrics, ticker, strategy)

        st.markdown("#### Rating & Verdict")
        st.write(f"**Rating:** {res.get('rating')} | **Risk Profile:** {res.get('risk')}")
        st.markdown(res.get("executive_summary", ""))

        with st.expander("Full Institutional Analysis JSON", expanded=False):
            st.json(res)
