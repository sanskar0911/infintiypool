"""
header.py

Top Header Component for the AI Quantitative Research Platform.
Displays Bloomberg-style top bar with live status indicator, environment badges, and presentation mode toggle.
"""

import textwrap
import streamlit as st

def render_top_header():
    """Render institutional top navigation header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        header_html = textwrap.dedent("""
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
                <div style="font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6, #60a5fa, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    QUANTUM LABS &bull; AI QUANT RESEARCH PLATFORM
                </div>
                <div style="display: flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; color: #10b981; font-weight: 600;">
                    <span style="height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block;"></span>
                    System Ready
                </div>
            </div>
            <div style="color: #94a3b8; font-size: 0.95rem; font-weight: 500; margin-bottom: 16px; display: flex; gap: 15px; align-items: center;">
                <span>📁 Dataset: <b>Indian Equity Universe (NSE)</b></span>
                <span>•</span>
                <span>📊 Benchmark: <b>NIFTY 50</b></span>
                <span>•</span>
                <span>⚡ Engine: <b>LangGraph Autonomous Quant Agent</b></span>
            </div>
        """)
        st.markdown(header_html, unsafe_allow_html=True)

    with col2:
        # Presentation mode toggle
        pres_mode = st.toggle("🎥 Presentation Mode", value=st.session_state.get("presentation_mode", False))
        if pres_mode != st.session_state.get("presentation_mode", False):
            st.session_state["presentation_mode"] = pres_mode
            st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0 0 15px 0;'>", unsafe_allow_html=True)
