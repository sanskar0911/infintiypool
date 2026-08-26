"""
ai_analysis_page.py

Page 11 — AI Strategy Analyst & Structured JSON Output Viewer.
Interfaces with LLM Strategy Analyzer to generate structured qualitative assessments.
"""

import json
import textwrap
import streamlit as st
from llm.analyzer import LLMStrategyAnalyzer

def render_ai_analysis_page(backtest_results: dict):
    """
    Render AI Analysis Page.
    """
    st.markdown("## 🤖 AI Investment Insights & Advisory")
    st.markdown("<div style='color: #94a3b8; font-size: 1.05rem; margin-bottom: 12px;'>LLM-driven qualitative strategy evaluation, risk assessment, and market regime recommendations.</div>", unsafe_allow_html=True)
    st.caption("⚠️ Disclaimer: AI analysis is interpretive and does not replace quantitative calculations.")

    if not backtest_results or "metrics" not in backtest_results:
        st.info("Please run a backtest to generate AI Strategy Analysis.")
        return

    metrics = backtest_results["metrics"]
    ticker = backtest_results.get("ticker", "INFY")
    strategy_name = backtest_results.get("strategy_name", "SMA")

    if "ai_analysis_res" not in st.session_state or st.session_state.get("ai_analysis_ticker") != ticker or st.session_state.get("ai_analysis_strat") != strategy_name:
        with st.spinner("Generating LLM Quantitative Assessment..."):
            try:
                analyzer = LLMStrategyAnalyzer()
                ai_res = analyzer.analyze_performance(metrics, ticker, strategy_name)
                st.session_state["ai_analysis_res"] = ai_res
                st.session_state["ai_analysis_ticker"] = ticker
                st.session_state["ai_analysis_strat"] = strategy_name
            except Exception as e:
                st.error(f"AI Analyst Error: {e}")
                return

    ai_res = st.session_state.get("ai_analysis_res", {})

    rating = ai_res.get("rating", "Neutral")
    risk_level = ai_res.get("risk", "Moderate")
    conf = ai_res.get("confidence", 0.85)

    emoji_map = {"Strong Buy": "🟢", "Moderate Buy": "🔵", "Neutral": "🟡", "Underperform": "🔴"}
    rating_emoji = emoji_map.get(rating, "⚪")

    badge_html = textwrap.dedent(f"""
        <div style="
            background: rgba(22, 28, 45, 0.9);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 0.75rem;
            padding: 1.2rem 1.5rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">
                    {rating_emoji} Institutional Rating: <span style="color: #3b82f6;">{rating}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                    Risk Level: <b>{risk_level}</b> | Confidence Score: <b>{conf*100:.0f}%</b>
                </div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); padding: 6px 14px; border-radius: 12px; font-weight: 700; font-size: 0.85rem;">
                🤖 AI Quant Agent
            </div>
        </div>
    """)
    st.markdown(badge_html, unsafe_allow_html=True)

    tab_eval, tab_json = st.tabs(["📋 Qualitative Evaluation", "🔍 View Structured JSON Output"])

    with tab_eval:
        if ai_res.get("executive_summary"):
            st.markdown("### 📋 Executive Summary")
            st.info(ai_res["executive_summary"])

        if ai_res.get("recommendation"):
            st.markdown("### 📌 Tactical Recommendation")
            st.warning(ai_res["recommendation"])

        st.markdown("---")
        st.markdown("### 📊 Metric-by-Metric Deep Dive")

        if ai_res.get("return_analysis"):
            with st.expander("📈 Return & CAGR Deep Dive", expanded=True):
                st.markdown(ai_res["return_analysis"])

        if ai_res.get("sharpe_analysis"):
            with st.expander("秤 Sharpe & Risk-Adjusted Efficiency", expanded=True):
                st.markdown(ai_res["sharpe_analysis"])

        if ai_res.get("drawdown_analysis"):
            with st.expander("📉 Tail Risk & Maximum Drawdown Recovery", expanded=True):
                st.markdown(ai_res["drawdown_analysis"])

    with tab_json:
        st.markdown("### 🔍 Raw Structured JSON Schema")
        st.markdown("Demonstrates structured JSON output parsing for agent workflows.")
        st.json(ai_res)
