"""
watchlist_page.py

Page 3 — Watchlist Interface for QUANTLAB.
"""

import os
import pandas as pd
import streamlit as st
from strategy import generate_signals


def render_watchlist_page():
    st.markdown("## ⭐ Custom Watchlist")
    st.caption("Personalized Equity Monitoring & Signal Tracking")

    if "watchlist_items" not in st.session_state:
        st.session_state["watchlist_items"] = ["INFY", "RELIANCE", "TCS"]

    watchlist = st.session_state["watchlist_items"]

    col_add, col_remove = st.columns(2)
    all_tickers = ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"]

    with col_add:
        add_ticker = st.selectbox("Add Stock to Watchlist", [t for t in all_tickers if t not in watchlist])
        if st.button("➕ Add Stock", use_container_width=True):
            if add_ticker and add_ticker not in watchlist:
                watchlist.append(add_ticker)
                st.session_state["watchlist_items"] = watchlist
                st.success(f"Added {add_ticker} to watchlist.")
                st.rerun()

    with col_remove:
        rem_ticker = st.selectbox("Remove Stock from Watchlist", watchlist if watchlist else ["None"])
        if st.button("❌ Remove Stock", use_container_width=True):
            if rem_ticker in watchlist:
                watchlist.remove(rem_ticker)
                st.session_state["watchlist_items"] = watchlist
                st.success(f"Removed {rem_ticker} from watchlist.")
                st.rerun()

    st.markdown("---")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    w_rows = []

    for tk in watchlist:
        p_path = os.path.join(base_dir, "data", "processed", f"{tk}.csv")
        if os.path.exists(p_path):
            df_tk = pd.read_csv(p_path, index_col="Date")
            df_tk.index = pd.to_datetime(df_tk.index)

            last_price = df_tk["Close"].iloc[-1]
            prev_price = df_tk["Close"].iloc[-2] if len(df_tk) > 1 else last_price
            change_pct = ((last_price - prev_price) / prev_price) * 100.0

            df_sma = generate_signals(df_tk, strategy_name="SMA", short_window=20, long_window=50)
            df_ema = generate_signals(df_tk, strategy_name="EMA", short_window=12, long_window=26)
            df_rsi = generate_signals(df_tk, strategy_name="RSI", period=14)

            sig_val = df_sma["Signal"].iloc[-1]
            sig_label = "🟢 BUY" if sig_val == 1 else ("🔴 SELL" if sig_val == -1 else "⚪ NEUTRAL")

            w_rows.append({
                "Stock": tk,
                "Price (₹)": f"₹{last_price:,.2f}",
                "Change (%)": f"{change_pct:+.2f}%",
                "SMA (20d)": f"₹{df_sma['SMA_20'].iloc[-1]:,.2f}",
                "EMA (12d)": f"₹{df_ema['EMA_12'].iloc[-1]:,.2f}",
                "RSI (14d)": f"{df_rsi['RSI'].iloc[-1]:.1f}",
                "Strategy Signal": sig_label,
                "Backtest Status": "Ready ✅"
            })

    if w_rows:
        st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Watchlist is currently empty. Add stocks above.")
