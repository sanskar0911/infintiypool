"""
overview_dashboard.py

Page 1 — Market & Research Dashboard for QUANTLAB.
"""

import os
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
from components.kpi_card import render_kpi_grid
from components.chart_container import render_equity_curve_plotly
from strategy import generate_signals


def render_overview_dashboard(backtest_results: dict):
    st.markdown("## 📊 Market & Research Dashboard")
    st.caption("QUANTLAB — AI-Powered Equity Backtesting & Research Terminal")

    metrics = backtest_results.get("metrics", {})
    bench_comp = backtest_results.get("bench_comp", {})
    portfolio_df = backtest_results.get("portfolio_df")
    benchmark_df = backtest_results.get("benchmark_df")
    ticker = backtest_results.get("ticker", "INFY")
    strategy = backtest_results.get("strategy_name", "SMA")

    # 1. Portfolio / Backtest Snapshot
    st.markdown("### 💼 Active Backtest Snapshot")
    if metrics:
        render_kpi_grid(metrics, bench_comp)

    st.markdown("---")

    # 2. Market Overview & Stock Signals
    st.markdown("### 🌐 Market Overview & Strategy Signals")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tickers = ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"]
    market_data = []

    for tk in tickers:
        p_path = os.path.join(base_dir, "data", "processed", f"{tk}.csv")
        if os.path.exists(p_path):
            df_tk = pd.read_csv(p_path, index_col="Date")
            last_price = df_tk["Close"].iloc[-1]
            prev_price = df_tk["Close"].iloc[-2] if len(df_tk) > 1 else last_price
            change_pct = ((last_price - prev_price) / prev_price) * 100.0
            volume = df_tk["Volume"].iloc[-1] if "Volume" in df_tk.columns else 0

            # Generate latest signal
            sig_df = generate_signals(df_tk, strategy_name="SMA", short_window=20, long_window=50)
            latest_sig = sig_df["Signal"].iloc[-1]
            sig_label = "🟢 BUY" if latest_sig == 1 else ("🔴 SELL" if latest_sig == -1 else "⚪ NEUTRAL")

            market_data.append({
                "Stock": tk,
                "Last Price (₹)": f"₹{last_price:,.2f}",
                "Change (%)": f"{change_pct:+.2f}%",
                "Volume": f"{int(volume):,}",
                "SMA Signal": sig_label
            })

    if market_data:
        m_df = pd.DataFrame(market_data)
        st.dataframe(m_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Strategy vs Benchmark Chart
    st.markdown("### 📈 Strategy vs Benchmark Equity Curve")
    col_chart, col_experiments = st.columns([2, 1])

    with col_chart:
        if portfolio_df is not None:
            render_equity_curve_plotly(
                portfolio_df,
                benchmark_df=benchmark_df,
                ticker=ticker,
                benchmark_symbol=backtest_results.get("benchmark", "NIFTY50")
            )

    with col_experiments:
        st.markdown("### 🔬 Recent Experiments")
        exp_data = [
            {"ID": "EXP_001", "Strategy": "SMA (20,50)", "Sharpe": 0.88, "Status": "Accepted"},
            {"ID": "EXP_002", "Strategy": "EMA (12,26)", "Sharpe": 0.91, "Status": "Validated"},
            {"ID": "EXP_003", "Strategy": "RSI (14,30/70)", "Sharpe": 0.72, "Status": "Historical"}
        ]
        st.dataframe(pd.DataFrame(exp_data), use_container_width=True, hide_index=True)

        st.markdown("### 🔄 Recent Agent Runs")
        agent_data = [
            {"Run": "Run #12", "Ticker": "INFY", "Node": "Research Evaluator", "Decision": "ACCEPT"},
            {"Run": "Run #11", "Ticker": "RELIANCE", "Node": "Backtest Tool", "Decision": "ITERATE"}
        ]
        st.dataframe(pd.DataFrame(agent_data), use_container_width=True, hide_index=True)
