"""
markets_page.py

Page 2 — Markets Terminal for QUANTLAB.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from strategy import generate_signals


def render_markets_page():
    st.markdown("## 🌐 Markets Research Terminal")
    st.caption("Real-Time Technical Indicator & Signal Monitor across Indian Equities")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tickers = ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"]

    search_term = st.text_input("🔍 Search Stock Universe", placeholder="e.g. RELIANCE, INFY...")

    market_rows = []
    for tk in tickers:
        if search_term and search_term.upper() not in tk.upper():
            continue

        p_path = os.path.join(base_dir, "data", "processed", f"{tk}.csv")
        if os.path.exists(p_path):
            df_tk = pd.read_csv(p_path, index_col="Date")
            df_tk.index = pd.to_datetime(df_tk.index)

            last_price = df_tk["Close"].iloc[-1]
            prev_price = df_tk["Close"].iloc[-2] if len(df_tk) > 1 else last_price
            change_pct = ((last_price - prev_price) / prev_price) * 100.0
            volume = df_tk["Volume"].iloc[-1] if "Volume" in df_tk.columns else 0

            # Generate SMA, EMA, RSI indicators
            df_sma = generate_signals(df_tk, strategy_name="SMA", short_window=20, long_window=50)
            df_ema = generate_signals(df_tk, strategy_name="EMA", short_window=12, long_window=26)
            df_rsi = generate_signals(df_tk, strategy_name="RSI", period=14)

            sma_val = df_sma["SMA_20"].iloc[-1] if "SMA_20" in df_sma.columns else 0.0
            ema_val = df_ema["EMA_12"].iloc[-1] if "EMA_12" in df_ema.columns else 0.0
            rsi_val = df_rsi["RSI"].iloc[-1] if "RSI" in df_rsi.columns else 50.0

            sig_val = df_sma["Signal"].iloc[-1]
            sig_label = "🟢 BUY" if sig_val == 1 else ("🔴 SELL" if sig_val == -1 else "⚪ NEUTRAL")

            market_rows.append({
                "Stock": tk,
                "Price (₹)": last_price,
                "Change (%)": change_pct,
                "Volume": volume,
                "SMA (20d)": sma_val,
                "EMA (12d)": ema_val,
                "RSI (14d)": rsi_val,
                "Signal": sig_label
            })

    if market_rows:
        df_display = pd.DataFrame(market_rows)
        
        # Formatted dataframe for display
        df_formatted = df_display.copy()
        df_formatted["Price (₹)"] = df_formatted["Price (₹)"].apply(lambda x: f"₹{x:,.2f}")
        df_formatted["Change (%)"] = df_formatted["Change (%)"].apply(lambda x: f"{x:+.2f}%")
        df_formatted["Volume"] = df_formatted["Volume"].apply(lambda x: f"{int(x):,}")
        df_formatted["SMA (20d)"] = df_formatted["SMA (20d)"].apply(lambda x: f"₹{x:,.2f}")
        df_formatted["EMA (12d)"] = df_formatted["EMA (12d)"].apply(lambda x: f"₹{x:,.2f}")
        df_formatted["RSI (14d)"] = df_formatted["RSI (14d)"].apply(lambda x: f"{x:.1f}")

        st.dataframe(df_formatted, use_container_width=True, hide_index=True)

        st.markdown("### 🔍 Jump to Stock Research")
        selected_stock = st.selectbox("Select stock to inspect:", tickers, index=0)
        if st.button(f"📊 Open {selected_stock} Research Page", type="primary"):
            st.session_state["selected_stock_research"] = selected_stock
            st.session_state["current_page"] = "Stock Research"
            st.rerun()
