"""
workflow_visualizer.py

LangGraph Visual State Graph & Research Loop Component.
Visualizes the autonomous quant research agent loop, active state node,
iteration counts, and loop decision state machine.
"""

import textwrap
import pandas as pd
import streamlit as st

def render_workflow_diagram(active_node: str = "research_evaluator", current_iteration: int = 1, loop_decision: str = "VALIDATE"):
    """
    Render visual flowchart node representation of the LangGraph research workflow.
    """
    nodes = [
        ("Load Data", "load_data", "📂"),
        ("Generate Signals", "run_backtest", "⚡"),
        ("Backtest Simulation", "run_backtest", "⚙️"),
        ("Calculate Metrics", "calculate_metrics", "📊"),
        ("Research Evaluator", "research_evaluator", "🔬"),
        ("Validation & OOS", "run_validation", "🛡️"),
        ("Fact Sheet & Report", "generate_artifacts", "📄")
    ]

    cols = st.columns(len(nodes))
    for idx, (label, node_id, icon) in enumerate(nodes):
        is_active = (node_id == active_node)
        border_color = "#10b981" if is_active else "rgba(255,255,255,0.1)"
        bg_color = "rgba(16, 185, 129, 0.2)" if is_active else "rgba(22, 28, 45, 0.7)"
        text_color = "#10b981" if is_active else "#94a3b8"
        
        with cols[idx]:
            node_html = textwrap.dedent(f"""
                <div style="
                    background: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: 0.6rem;
                    padding: 10px 6px;
                    text-align: center;
                    backdrop-filter: blur(6px);
                ">
                    <div style="font-size: 1.2rem; margin-bottom: 2px;">{icon}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: {text_color}; text-transform: uppercase;">{label}</div>
                </div>
            """)
            st.markdown(node_html, unsafe_allow_html=True)
            if idx < len(nodes) - 1:
                st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>→</div>", unsafe_allow_html=True)


def render_loop_engineering_breakdown(experiment_history: list):
    """
    Render step-by-step Research Loop Iteration breakdown timeline.
    """
    st.markdown("### 🔄 Research Loop Iteration Breakdown")

    if not experiment_history:
        st.info("No research loop iterations recorded yet. Run the LangGraph workflow to populate loop history.")
        return

    for item in experiment_history:
        iter_num = item.get("iteration", 1)
        strat = item.get("strategy", "Strategy")
        sharpe = item.get("sharpe", 0.0)
        cagr = item.get("cagr", 0.0) * 100.0
        excess = item.get("excess_return", 0.0) * 100.0
        decision = item.get("decision", item.get("loop_decision", "EVALUATE"))
        params_str = ", ".join([f"{k}={v}" for k, v in item.get("params", {}).items()])

        decision_color = "#10b981" if decision in ["ACCEPT", "VALIDATE", "PASS"] else ("#f59e0b" if decision == "ITERATE" else "#ef4444")

        item_html = textwrap.dedent(f"""
            <div style="
                background: rgba(15, 23, 42, 0.8);
                border-left: 4px solid {decision_color};
                border-radius: 0.5rem;
                padding: 12px 16px;
                margin-bottom: 10px;
                border-top: 1px solid rgba(255,255,255,0.05);
                border-right: 1px solid rgba(255,255,255,0.05);
                border-bottom: 1px solid rgba(255,255,255,0.05);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">
                        Iteration {iter_num} &bull; <span style="color: #60a5fa;">{strat}</span> ({params_str})
                    </div>
                    <div style="background: rgba(255,255,255,0.05); color: {decision_color}; padding: 2px 10px; border-radius: 10px; font-size: 0.8rem; font-weight: 700;">
                        Decision: {decision}
                    </div>
                </div>
                <div style="display: flex; gap: 20px; margin-top: 6px; font-size: 0.85rem; color: #94a3b8;">
                    <span>Sharpe Ratio: <b style="color: #f8fafc;">{sharpe:.2f}</b></span>
                    <span>CAGR: <b style="color: #f8fafc;">{cagr:+.2f}%</b></span>
                    <span>Excess Return: <b style="color: #f8fafc;">{excess:+.2f}%</b></span>
                </div>
            </div>
        """)
        st.markdown(item_html, unsafe_allow_html=True)
