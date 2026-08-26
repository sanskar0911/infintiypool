"""
navbar.py

Top Navigation Bar & Market Ticker Strip Component for QUANTLAB.
"""

import textwrap
import streamlit as st


def render_navbar():
    """Render top header bar with branding, market strip, global search, and presentation toggle."""
    # Top Ticker Strip
    ticker_strip_html = textwrap.dedent("""
        <div style="
            background: #0d1322;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding: 6px 16px;
            font-size: 0.82rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #94a3b8;
            margin-bottom: 12px;
        ">
            <div style="display: flex; gap: 18px; align-items: center;">
                <span style="font-weight: 700; color: #3b82f6;">MARKETS:</span>
                <span><b>NIFTY 50:</b> <span style="color: #10b981;">22,450.15 (+0.65%)</span></span>
                <span><b>SENSEX:</b> <span style="color: #10b981;">73,880.20 (+0.58%)</span></span>
                <span><b>BANK NIFTY:</b> <span style="color: #10b981;">47,520.10 (+0.42%)</span></span>
                <span><b>INFY:</b> <span style="color: #10b981;">₹1,605.40 (+1.2%)</span></span>
                <span><b>RELIANCE:</b> <span style="color: #ef4444;">₹2,980.10 (-0.4%)</span></span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b;">
                Shankh / Decuple Internship TE1 | Status: <span style="color: #10b981;">● Online</span>
            </div>
        </div>
    """)
    st.markdown(ticker_strip_html, unsafe_allow_html=True)

    col_title, col_search, col_toggle = st.columns([3, 2, 1])

    with col_title:
        title_html = textwrap.dedent("""
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #3b82f6; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: 900; color: white;">Q</div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 900; letter-spacing: 0.5px; color: #ffffff; line-height: 1.1;">
                        QUANT<span style="color: #3b82f6;">LAB</span>
                    </div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">
                        AI-Powered Equity Backtesting & Research Terminal
                    </div>
                </div>
            </div>
        """)
        st.markdown(title_html, unsafe_allow_html=True)

    with col_search:
        search_query = st.text_input(
            "Global Search",
            placeholder="🔍 Search stocks (INFY, RELIANCE), strategies, RAG docs...",
            label_visibility="collapsed"
        )
        if search_query:
            st.session_state["global_search_query"] = search_query

    with col_toggle:
        pres_active = st.session_state.get("presentation_mode", False)
        btn_label = "⏹️ Exit Pres. Mode" if pres_active else "🎬 Presentation Mode"
        if st.button(btn_label, use_container_width=True):
            st.session_state["presentation_mode"] = not pres_active
            st.rerun()
