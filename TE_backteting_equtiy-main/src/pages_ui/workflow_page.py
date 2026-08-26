"""
workflow_page.py

Page 10 — LangGraph Agent Workflow & Loop Engineering UI Renderer.
Visualizes the autonomous research state graph, current active state, iteration history,
and loop decision logic for internship presentation demonstrations.
"""

import textwrap
import pandas as pd
import streamlit as st
from agent.workflow import run_agent_workflow
from components.workflow_visualizer import render_workflow_diagram, render_loop_engineering_breakdown

def render_workflow_page(backtest_results: dict):
    """
    Render LangGraph Agent Workflow & Loop Engineering UI.
    """
    st.markdown("## 🔄 LangGraph Agent Workflow & Research Loop")
    st.markdown("<div style='color: #94a3b8; font-size: 1.05rem; margin-bottom: 20px;'>Autonomous LangGraph Quant Research State Machine with Controlled Loop Engineering.</div>", unsafe_allow_html=True)

    col_ctrl, col_info = st.columns([2, 1])

    with col_ctrl:
        st.markdown("### ⚙️ Autonomous Research Loop Parameters")
        ticker = st.selectbox("Ticker for Agent Search", ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"], index=0, key="wf_ticker")
        strategy = st.selectbox("Strategy Architecture", ["SMA", "EMA", "RSI"], index=0, key="wf_strat")
        max_iters = st.slider("Max Loop Iteration Cap", min_value=1, max_value=5, value=3, key="wf_max_iters")
        
        run_agent_btn = st.button("▶ Execute LangGraph Agent Loop", type="primary", use_container_width=True)

    with col_info:
        info_html = textwrap.dedent("""
            <div style="background: rgba(22, 28, 45, 0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.75rem; padding: 1.2rem; font-size: 0.9rem; line-height: 1.6;">
                <div style="font-weight: 700; color: #3b82f6; margin-bottom: 6px;">🧠 Research Loop Mechanics</div>
                <div>• <b>Entry:</b> Loads OHLCV data & initial hypothesis</div>
                <div>• <b>Evaluator:</b> Checks if Sharpe ≥ 1.0 & excess return > 0</div>
                <div>• <b>Loop Decision:</b> Triggers parameter adjustment if target missed</div>
                <div>• <b>Stopping Criteria:</b> Maximum iteration cap or validation pass</div>
            </div>
        """)
        st.markdown(info_html, unsafe_allow_html=True)

    st.markdown("---")

    if run_agent_btn or "agent_final_state" in st.session_state:
        if run_agent_btn:
            with st.spinner("Executing LangGraph Agent Research Loop across state nodes..."):
                final_state = run_agent_workflow(
                    ticker=ticker,
                    strategy_name=strategy,
                    max_iterations=max_iters
                )
                st.session_state["agent_final_state"] = final_state
                
                # Inter-page synchronization: Sync Agent results to global backtest_results store
                if final_state and final_state.get("metrics"):
                    st.session_state["backtest_results"] = {
                        "ticker": final_state.get("ticker", ticker),
                        "benchmark": final_state.get("benchmark_symbol", "NIFTY50"),
                        "strategy_name": final_state.get("strategy_name", strategy),
                        "config": {
                            "ticker": final_state.get("ticker", ticker),
                            "benchmark": final_state.get("benchmark_symbol", "NIFTY50"),
                            "strategy_name": final_state.get("strategy_name", strategy),
                            "initial_capital": final_state.get("initial_capital", 100000.0)
                        },
                        "signals_df": final_state.get("signals_df"),
                        "portfolio_df": final_state.get("portfolio_df"),
                        "trade_log_df": final_state.get("trade_log_df"),
                        "metrics": final_state.get("metrics"),
                        "benchmark_df": final_state.get("benchmark_df"),
                        "bench_comp": final_state.get("benchmark_comparison"),
                        "warnings_list": final_state.get("warnings", [])
                    }

        final_state = st.session_state.get("agent_final_state")
        if final_state:
            st.markdown("### 🗺️ State Node Pipeline Visualizer")
            active_node = "generate_artifacts" if final_state.get("loop_decision") in ["ACCEPT", "VALIDATE", "STOP"] else "research_evaluator"
            render_workflow_diagram(
                active_node=active_node,
                current_iteration=final_state.get("iteration", 1),
                loop_decision=final_state.get("loop_decision", "PENDING")
            )

            st.markdown("<br>", unsafe_allow_html=True)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Completed Iterations", final_state.get("iteration", 1))
            col_m2.metric("Final Loop Decision", final_state.get("loop_decision", "PENDING"))
            col_m3.metric("Total Warnings Raised", len(final_state.get("warnings", [])))
            col_m4.metric("Stopping Condition", "Max Iterations Reached" if final_state.get("iteration", 1) >= max_iters else "Target Sharpe Met")

            st.markdown("---")

            history = final_state.get("experiment_history", [])
            render_loop_engineering_breakdown(history)

            st.markdown("---")
            st.markdown("### 🔗 Synchronized Agent Action Hub")
            col_a1, col_a2, col_a3 = st.columns(3)

            if col_a1.button("📈 View Agent Results", use_container_width=True):
                st.session_state["current_page"] = "Results"
                st.rerun()

            if col_a2.button("📄 Generate Agent Fact Sheet", use_container_width=True):
                st.session_state["current_page"] = "Reports"
                st.rerun()

            if col_a3.button("🤖 AI Quantitative Analysis", use_container_width=True):
                st.session_state["current_page"] = "AI Analysis"
                st.rerun()

    else:
        st.markdown("### 🗺️ State Node Pipeline Visualizer (Standby)")
        render_workflow_diagram(active_node="research_evaluator", current_iteration=1, loop_decision="STANDBY")
        st.info("Click **▶ Execute LangGraph Agent Loop** to run the live research loop.")
