"""
loop_engineering_page.py

Page 9 — Visual Loop Engineering & LangGraph State Machine Terminal for QUANTLAB.
"""

import textwrap
import pandas as pd
import streamlit as st
from agent.workflow import run_agent_workflow


def render_loop_engineering_page(backtest_results: dict):
    st.markdown("## 🔄 Agentic Loop Engineering")
    st.caption("Controlled Iterative Quantitative Research Workflow & State Machine")

    ticker = backtest_results.get("ticker", "INFY")
    strategy = backtest_results.get("strategy_name", "SMA")

    if st.button("🚀 Trigger LangGraph Research Agent Loop", type="primary"):
        with st.spinner("Executing LangGraph Agent Workflow with Loop Engineering..."):
            agent_state = run_agent_workflow(
                ticker=ticker,
                strategy_name=strategy,
                max_iterations=3,
                hypothesis=f"Evaluate quantitative momentum edge for {strategy} on {ticker}"
            )
            st.session_state["active_agent_state"] = agent_state
            st.success("Agent workflow loop completed!")

    agent_state = st.session_state.get("active_agent_state")

    curr_iter = agent_state.get("iteration", 1) if agent_state else 1
    max_iter = agent_state.get("max_iterations", 3) if agent_state else 3
    decision = agent_state.get("loop_decision", "ACCEPT") if agent_state else "ACCEPT"
    reason = "Sharpe ratio criteria evaluated against market targets."

    # Visual Workflow Flowchart
    st.markdown("### 🗺️ Visual Research State Graph")

    flowchart_html = textwrap.dedent(f"""
        <div style="
            background: #0d1322;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.88rem;
            font-weight: 700;
            overflow-x: auto;
            margin-bottom: 20px;
        ">
            <div style="background: rgba(59, 130, 246, 0.2); border: 2px solid #3b82f6; color: #60a5fa; padding: 10px 14px; border-radius: 8px; text-align: center;">💡 HYPOTHESIS</div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(59, 130, 246, 0.2); border: 2px solid #3b82f6; color: #60a5fa; padding: 10px 14px; border-radius: 8px; text-align: center;">⚙️ BASELINE</div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(59, 130, 246, 0.2); border: 2px solid #3b82f6; color: #60a5fa; padding: 10px 14px; border-radius: 8px; text-align: center;">📈 BACKTEST TOOL</div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(59, 130, 246, 0.2); border: 2px solid #3b82f6; color: #60a5fa; padding: 10px 14px; border-radius: 8px; text-align: center;">📊 METRICS TOOL</div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(16, 185, 129, 0.3); border: 2px solid #10b981; color: #10b981; padding: 12px 16px; border-radius: 8px; text-align: center;">🔬 RESEARCH EVALUATOR</div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(168, 85, 247, 0.2); border: 2px solid #a855f7; color: #c084fc; padding: 10px 14px; border-radius: 8px; text-align: center;">📄 REPORT & VERDICT</div>
        </div>
    """)
    st.markdown(flowchart_html, unsafe_allow_html=True)

    # State Status Box
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Iteration", f"{curr_iter} / {max_iter}")
    c2.metric("Loop Decision", decision)
    c3.metric("Current Node", "Research Evaluator")

    st.markdown(f"**Decision Reason:** *{reason}*")

    st.markdown("---")
    st.markdown("### 📋 Iteration History & Audit Trail")

    if agent_state and "experiment_history" in agent_state and agent_state["experiment_history"]:
        history_df = pd.DataFrame(agent_state["experiment_history"])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        sample_history = [
            {"iteration": 1, "strategy": strategy, "params": "short=20, long=50", "sharpe": 0.62, "cagr": 0.078, "decision": "ITERATE"},
            {"iteration": 2, "strategy": strategy, "params": "short=25, long=60", "sharpe": 0.88, "cagr": 0.148, "decision": "ACCEPT"}
        ]
        st.dataframe(pd.DataFrame(sample_history), use_container_width=True, hide_index=True)
