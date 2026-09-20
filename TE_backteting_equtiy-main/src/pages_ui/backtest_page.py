"""
backtest_page.py

Page 6 — Backtest Configuration & Interactive Simulation Terminal for QUANTLAB.
Supports complete Execution Layer settings: Order Timing, Slippage bps, Commission bps,
Position Sizing, Cash Constraints, and Short Selling policies.
"""

import textwrap
import streamlit as st


def render_backtest_page(config: dict, execute_callback):
    st.markdown("## ⚙️ Backtest Configuration Terminal")
    st.caption("Side-by-side Configuration & Upgraded Execution Simulation Engine")

    col_config, col_preview = st.columns([1, 1])

    with col_config:
        st.markdown("### 🛠️ BACKTEST CONFIGURATION")

        st.markdown("#### 1. Data & Asset Universe")
        ticker = st.selectbox("Select Stock", ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"], index=0)
        benchmark = st.selectbox("Select Benchmark", ["NIFTY50"], index=0)

        st.markdown("#### 2. Strategy Architecture")
        strategy_name = st.selectbox("Select Strategy", ["SMA", "EMA", "RSI"], index=0)

        strategy_kwargs = {}
        if strategy_name == "SMA":
            short_w = st.number_input("Short Window", min_value=5, max_value=100, value=20)
            long_w = st.number_input("Long Window", min_value=10, max_value=200, value=50)
            strategy_kwargs = {"short_window": short_w, "long_window": long_w}
        elif strategy_name == "EMA":
            short_w = st.number_input("Short Window", min_value=5, max_value=100, value=12)
            long_w = st.number_input("Long Window", min_value=10, max_value=200, value=26)
            strategy_kwargs = {"short_window": short_w, "long_window": long_w}
        elif strategy_name == "RSI":
            rsi_p = st.number_input("RSI Period", min_value=5, max_value=50, value=14)
            oversold = st.number_input("Oversold Level", min_value=10.0, max_value=45.0, value=30.0)
            overbought = st.number_input("Overbought Level", min_value=55.0, max_value=90.0, value=70.0)
            strategy_kwargs = {"period": rsi_p, "oversold": oversold, "overbought": overbought}

        st.markdown("#### 3. Execution Engine & Friction Controls")
        initial_capital = st.number_input("Initial Capital (INR)", min_value=10000.0, value=100000.0, step=10000.0)
        position_size = st.slider("Capital Allocation (%)", min_value=10, max_value=100, value=20, step=5) / 100.0

        execution_timing = st.selectbox(
            "Execution Timing",
            ["NEXT_BAR_OPEN", "SAME_BAR_CLOSE", "NEXT_BAR_CLOSE"],
            index=0,
            help="NEXT_BAR_OPEN eliminates look-ahead bias by executing orders on next bar's Open price."
        )

        slippage_bps = st.number_input("Slippage (basis points)", min_value=0.0, max_value=200.0, value=10.0, step=1.0, help="10 bps = 0.10% price penalty")
        commission_bps = st.number_input("Commission (basis points)", min_value=0.0, max_value=200.0, value=5.0, step=1.0, help="5 bps = 0.05% brokerage fee")

        insufficient_cash = st.selectbox(
            "Insufficient Cash Policy",
            ["REDUCE_QUANTITY", "REJECT"],
            index=0,
            help="REDUCE_QUANTITY resizes trade to affordable cash. REJECT cancels order."
        )

        allow_short = st.selectbox("Allow Short Selling", ["OFF", "ON"], index=0) == "ON"

        run_clicked = st.button("🚀 RUN BACKTEST ENGINE", type="primary", use_container_width=True)

    with col_preview:
        st.markdown("### 📋 PREVIEW & EXECUTION SUMMARY")

        preview_html = textwrap.dedent(f"""
            <div style="
                background: #111827;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 18px;
                line-height: 1.8;
                margin-bottom: 16px;
            ">
                <div>📌 <b>Target Asset:</b> <span style="color: #60a5fa;">{ticker}</span></div>
                <div>🎯 <b>Benchmark:</b> <span style="color: #60a5fa;">{benchmark}</span></div>
                <div>⚙️ <b>Strategy Architecture:</b> <span style="color: #60a5fa;">{strategy_name}</span></div>
                <div>💵 <b>Initial Capital:</b> ₹{initial_capital:,.2f}</div>
                <div>📊 <b>Position Sizing:</b> {position_size*100:.0f}%</div>
                <div>⏱️ <b>Timing Mode:</b> <span style="color: #34d399;">{execution_timing}</span></div>
                <div>💸 <b>Friction:</b> Comm {commission_bps} bps ({commission_bps/100:.2f}%) | Slip {slippage_bps} bps ({slippage_bps/100:.2f}%)</div>
                <div>🛡️ <b>Cash Policy:</b> {insufficient_cash} | <b>Shorting:</b> {'Allowed' if allow_short else 'Disabled'}</div>
            </div>
        """)
        st.markdown(preview_html, unsafe_allow_html=True)

        if run_clicked:
            new_config = {
                "ticker": ticker,
                "benchmark": benchmark,
                "strategy_name": strategy_name,
                "strategy_kwargs": strategy_kwargs,
                "initial_capital": initial_capital,
                "position_size": position_size,
                "commission_pct": commission_bps / 10000.0,
                "slippage_pct": slippage_bps / 10000.0,
                "commission_bps": commission_bps,
                "slippage_bps": slippage_bps,
                "execution_timing": execution_timing,
                "insufficient_cash_policy": insufficient_cash,
                "allow_short_selling": allow_short,
                "run_triggered": True
            }
            execute_callback(new_config)
            st.success("Backtest simulation completed! Check Backtest Results page.")

        backtest_results = st.session_state.get("backtest_results", {})
        if backtest_results and "metrics" in backtest_results:
            m = backtest_results["metrics"]
            st.markdown("#### Latest Simulation Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("CAGR", f"{m.get('CAGR', 0)*100:.2f}%")
            c2.metric("Sharpe Ratio", f"{m.get('Sharpe Ratio', 0):.2f}")
            c3.metric("Max Drawdown", f"{m.get('Maximum Drawdown', 0)*100:.2f}%")
