"""
presentation_mode.py

Sequential 10-Step Presentation Mode Controller for Internship Learning Journey.
(11-Jul to 10-Oct Milestones)
"""

import os
import streamlit as st
from pages_ui.overview_dashboard import render_overview_dashboard
from pages_ui.backtest_page import render_backtest_page
from pages_ui.benchmark_page import render_benchmark_page
from pages_ui.ai_analysis_page import render_ai_analysis_page
from pages_ui.workflow_page import render_workflow_page
from pages_ui.rag_page import render_rag_page
from pages_ui.memory_page import render_memory_page
from pages_ui.automation_page import render_automation_page
from pages_ui.about_page import render_about_page


def render_presentation_mode(backtest_results: dict):
    """Render sequential 10-step internship learning journey presentation mode."""
    st.markdown("""
        <div style="background: linear-gradient(90deg, #1e293b, #0f172a); padding: 14px 20px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 20px;">
            <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc;">
                🎬 PRESENTATION MODE — SHANKH / DECUPLE INTERNSHIP 2026-27 PROJECT TE1
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                Sequential Walkthrough of Learning Milestones (11-Jul-2026 to 10-Oct-2026)
            </div>
        </div>
    """, unsafe_allow_html=True)

    milestones = [
        "01 — Python & Backtesting (11-Jul)",
        "02 — AI Coding (25-Jul)",
        "03 — LLM (01-Aug)",
        "04 — Agentic Workflow (15-Aug)",
        "05 — Loop Engineering (15-Aug / Core)",
        "06 — RAG (29-Aug)",
        "07 — Memory (29-Aug)",
        "08 — Automation (12-Sep)",
        "09 — Security & Deployment (26-Sep)",
        "10 — Delivery Baseline (10-Oct)"
    ]

    if "presentation_step" not in st.session_state:
        st.session_state["presentation_step"] = 0

    curr_idx = st.session_state["presentation_step"]

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(curr_idx == 0), use_container_width=True):
            st.session_state["presentation_step"] -= 1
            st.rerun()

    with col2:
        selected = st.selectbox(
            "Select Milestone Step",
            milestones,
            index=curr_idx,
            label_visibility="collapsed"
        )
        st.session_state["presentation_step"] = milestones.index(selected)
        curr_idx = st.session_state["presentation_step"]

    with col3:
        if st.button("Next ➡️", disabled=(curr_idx == len(milestones) - 1), use_container_width=True):
            st.session_state["presentation_step"] += 1
            st.rerun()

    st.markdown("---")

    # Step Routing
    if curr_idx == 0:
        st.subheader("Milestone 01: Python, Data & Backtesting Foundation (11-Jul-2026)")
        render_overview_dashboard(backtest_results)

    elif curr_idx == 1:
        st.subheader("Milestone 02: AI Coding Demonstration (25-Jul-2026)")
        doc_path = os.path.join("docs", "AI_CODING_DEMO.md")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.error("AI_CODING_DEMO.md not found.")

    elif curr_idx == 2:
        st.subheader("Milestone 03: LLM Foundations & Structured Output (01-Aug-2026)")
        render_ai_analysis_page(backtest_results)

    elif curr_idx == 3:
        st.subheader("Milestone 04: Agentic Workflow Foundations (15-Aug-2026)")
        render_workflow_page(backtest_results)

    elif curr_idx == 4:
        st.subheader("Milestone 05: Agentic Loop Engineering (15-Aug Core Feature)")
        doc_path = os.path.join("docs", "LOOP_ENGINEERING.md")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        render_workflow_page(backtest_results)

    elif curr_idx == 5:
        st.subheader("Milestone 06: RAG & FAISS Vector Store Assistant (29-Aug-2026)")
        render_rag_page()

    elif curr_idx == 6:
        st.subheader("Milestone 07: Conversation State Memory (29-Aug-2026)")
        render_memory_page()

    elif curr_idx == 7:
        st.subheader("Milestone 08: Webhook Automation & Integration (12-Sep-2026)")
        render_automation_page(backtest_results)

    elif curr_idx == 8:
        st.subheader("Milestone 09: Security, Logging & Deployment (26-Sep-2026)")
        doc_path = os.path.join("docs", "DEPLOYMENT.md")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    elif curr_idx == 9:
        st.subheader("Milestone 10: Delivery Baseline & Gap Report (10-Oct-2026)")
        doc_path = os.path.join("docs", "BASELINE_REPORT.md")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
