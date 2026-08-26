"""
experiments_page.py

Page 8 — Experiment Lab for QUANTLAB.
"""

import os
import pandas as pd
import streamlit as st
from strategy import generate_signals
from backtester import run_backtest
from metrics import calculate_metrics


def render_experiments_page(backtest_results: dict):
    st.markdown("## 🔬 Experiment Lab")
    st.caption("Hypothesis-Driven Quantitative Research & Controlled Validation Lab")

    if "experiment_history_list" not in st.session_state:
        st.session_state["experiment_history_list"] = [
            {
                "Experiment ID": "EXP_INFY_SMA_01",
                "Hypothesis": "Test SMA crossover momentum edge on INFY",
                "Strategy": "SMA",
                "Parameters": "short=20, long=50",
                "CAGR": "14.8%",
                "Sharpe": "0.88",
                "Drawdown": "-12.5%",
                "Validation Status": "Historical Result"
            },
            {
                "Experiment ID": "EXP_RELIANCE_EMA_02",
                "Hypothesis": "Test fast EMA momentum on RELIANCE",
                "Strategy": "EMA",
                "Parameters": "short=12, long=26",
                "CAGR": "16.2%",
                "Sharpe": "0.91",
                "Drawdown": "-14.2%",
                "Validation Status": "Validated"
            }
        ]

    with st.form("experiment_form"):
        st.markdown("### 🧪 Launch New Quantitative Experiment")
        exp_name = st.text_input("Experiment Name / ID", value="EXP_NEW_HYPOTHESIS_01")
        hypothesis = st.text_area("Research Hypothesis", value="Test if shortening RSI period improves mean reversion signals on large-cap Indian equities.")
        ticker = st.selectbox("Stock", ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"], index=0)
        strategy = st.selectbox("Strategy Architecture", ["SMA", "EMA", "RSI"], index=0)

        submitted = st.form_submit_button("🚀 RUN EXPERIMENT")

        if submitted:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            p_path = os.path.join(base_dir, "data", "processed", f"{ticker}.csv")

            if os.path.exists(p_path):
                df = pd.read_csv(p_path, index_col="Date")
                df.index = pd.to_datetime(df.index)
                sig_df = generate_signals(df, strategy_name=strategy)
                port_df, t_df = run_backtest(sig_df)
                m = calculate_metrics(port_df, t_df)

                new_entry = {
                    "Experiment ID": exp_name,
                    "Hypothesis": hypothesis,
                    "Strategy": f"{strategy} on {ticker}",
                    "Parameters": "Default Config",
                    "CAGR": f"{m['CAGR']*100:.1f}%",
                    "Sharpe": f"{m['Sharpe Ratio']:.2f}",
                    "Drawdown": f"{m['Maximum Drawdown']*100:.1f}%",
                    "Validation Status": "Historical Result"
                }

                st.session_state["experiment_history_list"].append(new_entry)
                st.success(f"Experiment {exp_name} completed! Result recorded.")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Experiment History & Audit Log")
    exp_df = pd.DataFrame(st.session_state["experiment_history_list"])
    st.dataframe(exp_df, use_container_width=True, hide_index=True)
