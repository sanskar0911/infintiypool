"""
stock_research_page.py

Page 4 — Professional Stock Detail & Research Terminal for QUANTLAB.
"""

import os
import textwrap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from strategy import generate_signals
from backtester import run_backtest
from metrics import calculate_metrics
from llm.analyzer import LLMStrategyAnalyzer
from rag.rag_chain import RAGChain


def render_stock_research_page(backtest_results: dict):
    selected_stock = st.session_state.get("selected_stock_research", "INFY")

    st.markdown(f"## 🏢 Stock Research: {selected_stock}")
    st.caption(f"Deep Quantitative Analysis, Technical Indicators, and AI Evaluation for {selected_stock}")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p_path = os.path.join(base_dir, "data", "processed", f"{selected_stock}.csv")

    if not os.path.exists(p_path):
        st.error(f"Price data file for {selected_stock} not found.")
        return

    df = pd.read_csv(p_path, index_col="Date")
    df.index = pd.to_datetime(df.index)

    last_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) > 1 else last_price
    change_pct = ((last_price - prev_price) / prev_price) * 100.0
    vol = df["Volume"].iloc[-1] if "Volume" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price (₹)", f"₹{last_price:,.2f}", f"{change_pct:+.2f}%")
    col2.metric("Daily Volume", f"{int(vol):,}")
    col3.metric("52-Week High (₹)", f"₹{df['Close'].max():,.2f}")
    col4.metric("52-Week Low (₹)", f"₹{df['Close'].min():,.2f}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Technicals", "Strategy", "Backtest", "Risk", "AI Research"])

    with tab1:
        st.markdown("### Price & Volume History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="#3b82f6", width=2)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Technical Indicators (SMA, EMA, RSI)")
        df_sig = generate_signals(df, strategy_name="SMA", short_window=20, long_window=50)
        df_rsi = generate_signals(df, strategy_name="RSI", period=14)

        fig_tech = go.Figure()
        fig_tech.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="#94a3b8")))
        if "SMA_20" in df_sig.columns:
            fig_tech.add_trace(go.Scatter(x=df.index, y=df_sig["SMA_20"], name="SMA 20d", line=dict(color="#10b981")))
        if "SMA_50" in df_sig.columns:
            fig_tech.add_trace(go.Scatter(x=df.index, y=df_sig["SMA_50"], name="SMA 50d", line=dict(color="#ef4444")))

        fig_tech.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig_tech, use_container_width=True)

    with tab3:
        st.markdown("### Strategy Signals & Parameters")
        st.info("Current active strategy: **SMA Crossover (20/50)**")
        df_sma = generate_signals(df, strategy_name="SMA", short_window=20, long_window=50)
        latest_sig = df_sma["Signal"].iloc[-1]
        st.metric("Latest Signal", "🟢 BUY" if latest_sig == 1 else ("🔴 SELL" if latest_sig == -1 else "⚪ NEUTRAL"))

    with tab4:
        st.markdown("### Backtest Results for " + selected_stock)
        df_sig = generate_signals(df, strategy_name="SMA")
        port_df, trade_df = run_backtest(df_sig)
        metrics = calculate_metrics(port_df, trade_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{metrics['Total Return']*100:.2f}%")
        c2.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
        c3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        c4.metric("Max Drawdown", f"{metrics['Maximum Drawdown']*100:.2f}%")

        if not trade_df.empty:
            st.markdown("#### Trade History Log")
            st.dataframe(trade_df.tail(10), use_container_width=True)

    with tab5:
        st.markdown("### Risk Analysis & Volatility Breakdown")
        df_sig = generate_signals(df, strategy_name="SMA")
        port_df, trade_df = run_backtest(df_sig)
        metrics = calculate_metrics(port_df, trade_df)

        st.json({
            "Annual Volatility": f"{metrics['Annual Volatility']*100:.2f}%",
            "Sortino Ratio": f"{metrics['Sortino Ratio']:.2f}",
            "Calmar Ratio": f"{metrics['Calmar Ratio']:.2f}",
            "Win Rate": f"{metrics['Win Rate']*100:.2f}%",
            "Profit Factor": f"{metrics['Profit Factor']:.2f}"
        })

    with tab6:
        st.markdown("### AI Research & RAG Context")
        analyzer = LLMStrategyAnalyzer()
        df_sig = generate_signals(df, strategy_name="SMA")
        port_df, trade_df = run_backtest(df_sig)
        metrics = calculate_metrics(port_df, trade_df)
        llm_res = analyzer.analyze_performance(metrics, selected_stock, "SMA")

        st.markdown("#### AI Executive Verdict")
        st.write(llm_res.get("executive_summary", ""))

        st.markdown("#### RAG Knowledge Retrieval")
        rag_chain = RAGChain()
        rag_res = rag_chain.answer_question(f"What is the analysis for {selected_stock} using SMA?")
        st.write(rag_res.get("answer", ""))
