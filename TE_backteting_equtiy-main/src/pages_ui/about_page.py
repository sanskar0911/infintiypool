"""
about_page.py

About & Final Requirements Matrix Page for Shankh / Decuple Internship 2026-27 Project TE1.
"""

import streamlit as st


def render_about_page():
    st.markdown("## ℹ️ About & Internship Learning Milestones Matrix")
    st.markdown("### Shankh / Decuple Internship 2026-27 Project TE1")

    st.info("Project: **Backtesting Equity Strategies** — Clean Presentation-Focused Build")

    st.markdown("### Requirements Matrix")

    matrix_data = [
        {"Internship Date": "11-Jul-2026", "Learning Topic": "Python / Git / Setup", "Project Evidence": "Repository, requirements, data loaders, backtester", "Files": "backtester.py, metrics.py, data_loader.py", "Status": "✅ Completed"},
        {"Internship Date": "25-Jul-2026", "Learning Topic": "AI Coding Tools", "Project Evidence": "AI code review, vulnerability analysis, refactored metrics", "Files": "docs/AI_CODING_DEMO.md, metrics.py", "Status": "✅ Completed"},
        {"Internship Date": "01-Aug-2026", "Learning Topic": "LLM Foundations", "Project Evidence": "Structured JSON rating, prompt templates, rule fallback", "Files": "src/llm/analyzer.py, prompts.py, LLM_DEMO.md", "Status": "✅ Completed"},
        {"Internship Date": "15-Aug-2026", "Learning Topic": "Agentic Workflows", "Project Evidence": "LangGraph StateGraph, tool calling, execution trace", "Files": "src/agent/workflow.py, graph.py, tools.py", "Status": "✅ Completed"},
        {"Internship Date": "15-Aug-2026", "Learning Topic": "Loop Engineering", "Project Evidence": "State machine evaluator, bounded iteration, max 3 loops", "Files": "src/agent/loop.py, docs/LOOP_ENGINEERING.md", "Status": "✅ Completed"},
        {"Internship Date": "29-Aug-2026", "Learning Topic": "RAG & Vector Store", "Project Evidence": "FAISS vector store, document chunker, semantic search", "Files": "src/rag/, knowledge_base/, RAG_ARCHITECTURE.md", "Status": "✅ Completed"},
        {"Internship Date": "29-Aug-2026", "Learning Topic": "Retrieval Evaluation", "Project Evidence": "Top-1 & Top-3 retrieval accuracy benchmarking", "Files": "src/rag/evaluation.py, RAG_EVALUATION.md", "Status": "✅ Completed"},
        {"Internship Date": "29-Aug-2026", "Learning Topic": "Conversation Memory", "Project Evidence": "Multi-turn chat context memory & [Clear Memory] reset", "Files": "src/rag/memory.py, docs/MEMORY.md", "Status": "✅ Completed"},
        {"Internship Date": "12-Sep-2026", "Learning Topic": "Automation & Webhooks", "Project Evidence": "Post-backtest webhook payload sender & handler", "Files": "automation/webhook_demo/, AUTOMATION.md", "Status": "✅ Completed"},
        {"Internship Date": "26-Sep-2026", "Learning Topic": "Deployment & Safety", "Project Evidence": ".env.example, logging setup, API safety, README", "Files": ".env.example, logs/app.log, DEPLOYMENT.md", "Status": "✅ Completed"},
        {"Internship Date": "10-Oct-2026", "Learning Topic": "Delivery Baseline", "Project Evidence": "Historical accuracy baseline, metrics gap analysis", "Files": "docs/BASELINE_REPORT.md, PRESENTATION_SCOPE.md", "Status": "✅ Completed"}
    ]

    st.table(matrix_data)

    st.markdown("---")
    st.markdown("### Preserved Archive Declaration")
    st.caption("Advanced quantitative research features (parameter sweeps, walk-forward testing, out-of-sample validation, regime analysis, liquidity filtering, and complex portfolio optimizers) have been preserved in the `archive/` directory and remote GitHub backup.")
