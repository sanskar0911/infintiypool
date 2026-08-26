"""
strategy_card.py

Strategy Library Visual Cards Component.
Displays Strategy Metadata, Formula Logic, Parameters, Best Use Cases,
and Quick Run Action Button.
"""

import textwrap
import streamlit as st

def render_strategy_card(
    name: str,
    description: str,
    parameters: dict,
    signal_logic: str,
    best_use_case: str,
    status: str = "Active",
    on_run_click: callable = None
):
    """
    Render a single strategy library card.
    """
    param_str = ", ".join([f"<b>{k}:</b> {v}" for k, v in parameters.items()])

    card_html = textwrap.dedent(f"""
        <div style="
            background: rgba(22, 28, 45, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 1.3rem; font-weight: 700; color: #3b82f6;">
                    ⚡ {name}
                </div>
                <div style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">
                    ● {status}
                </div>
            </div>
            <div style="color: #cbd5e1; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.4;">
                {description}
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 12px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 10px; font-size: 0.85rem; color: #94a3b8;">
                <div style="margin-bottom: 4px;">⚙️ <b>Parameters:</b> {param_str}</div>
                <div style="margin-bottom: 4px;">🧠 <b>Signal Logic:</b> <span style="color: #f8fafc;">{signal_logic}</span></div>
                <div>🎯 <b>Best Use Case:</b> <span style="color: #60a5fa;">{best_use_case}</span></div>
            </div>
        </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)
