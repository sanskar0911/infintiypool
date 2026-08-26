"""
sidebar.py

QUANTLAB Streamlined Left Navigation Sidebar Component.
"""

import textwrap
import streamlit as st
from strategies.factory import StrategyFactory


def render_sidebar():
    """
    Render QUANTLAB sidebar menu and backtest controls.
    Returns selected page and backtest config dict.
    """
    st.sidebar.markdown("""
        <div style="font-size: 1.15rem; font-weight: 800; color: #3b82f6; margin-bottom: 8px;">
            ⚡ QUANTLAB TERMINAL
        </div>
    """, unsafe_allow_html=True)

    pages = [
        "Dashboard",
        "Markets",
        "Stock Research",
        "Strategies",
        "AI Assistant",
        "RAG Research",
        "Reports"
    ]

    icons = {
        "Dashboard": "📊",
        "Markets": "🌐",
        "Stock Research": "🔍",
        "Strategies": "📚",
        "AI Assistant": "🤖",
        "RAG Research": "📖",
        "Reports": "📄"
    }

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        format_func=lambda x: f"{icons.get(x, '📄')} {x}",
        index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0
    )
    st.session_state["current_page"] = selected_page

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-bottom: 8px;'>⚙️ BACKTEST CONTROLS</div>", unsafe_allow_html=True)

    available_tickers = ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"]
    selected_ticker = st.sidebar.selectbox("Stock Universe", available_tickers, index=0)
    selected_benchmark = st.sidebar.selectbox("Benchmark", ["NIFTY50"], index=0)

    available_strategies = StrategyFactory.list_strategies()
    selected_strategy = st.sidebar.selectbox("Strategy Architecture", available_strategies, index=0)

    strategy_kwargs = {}
    if selected_strategy == "SMA":
        short_w = st.sidebar.number_input("Short Window", min_value=5, max_value=100, value=20)
        long_w = st.sidebar.number_input("Long Window", min_value=10, max_value=200, value=50)
        strategy_kwargs = {"short_window": short_w, "long_window": long_w}
    elif selected_strategy == "EMA":
        short_w = st.sidebar.number_input("Short Window", min_value=5, max_value=100, value=12)
        long_w = st.sidebar.number_input("Long Window", min_value=10, max_value=200, value=26)
        strategy_kwargs = {"short_window": short_w, "long_window": long_w}
    elif selected_strategy == "RSI":
        rsi_p = st.sidebar.number_input("RSI Period", min_value=5, max_value=50, value=14)
        oversold = st.sidebar.number_input("Oversold Level", min_value=10.0, max_value=45.0, value=30.0)
        overbought = st.sidebar.number_input("Overbought Level", min_value=55.0, max_value=90.0, value=70.0)
        strategy_kwargs = {"period": rsi_p, "oversold": oversold, "overbought": overbought}

    with st.sidebar.expander("💼 Capital & Risk Controls", expanded=False):
        initial_capital = st.number_input("Initial Capital (INR)", min_value=10000.0, value=100000.0, step=10000.0)
        commission_pct = st.slider("Commission (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100.0
        slippage_pct = st.slider("Slippage (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01) / 100.0
        position_size_pct = st.slider("Position Size (%)", min_value=10, max_value=100, value=20, step=5) / 100.0

        use_sl = st.checkbox("Enable Stop Loss", value=True)
        stop_loss_pct = (st.slider("Stop Loss (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100.0) if use_sl else None

        use_tp = st.checkbox("Enable Take Profit", value=True)
        take_profit_pct = (st.slider("Take Profit (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0) / 100.0) if use_tp else None

    run_backtest_clicked = st.sidebar.button("🚀 Run Backtest Engine", type="primary", use_container_width=True)

    config = {
        "ticker": selected_ticker,
        "benchmark": selected_benchmark,
        "strategy_name": selected_strategy,
        "strategy_kwargs": strategy_kwargs,
        "initial_capital": initial_capital,
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct,
        "position_size": position_size_pct,
        "stop_loss": stop_loss_pct,
        "take_profit": take_profit_pct,
        "run_triggered": run_backtest_clicked
    }

    return selected_page, config
