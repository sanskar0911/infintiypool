"""
rag_page.py

Streamlit page for FAISS RAG Knowledge Assistant & Empirical Retrieval Evaluation.
(29-Aug Milestone)
"""

import streamlit as st
from rag.rag_chain import RAGChain
from rag.evaluation import evaluate_rag_retrieval


def render_rag_page():
    st.markdown("## 📚 RAG Research Assistant & Retrieval Evaluation")
    st.caption("Demonstrating 29-Aug-2026 Milestone: Grounded Q&A via FAISS Vector Store over Markdown Knowledge Base")

    tab1, tab2 = st.tabs(["🔍 RAG Q&A Assistant", "📊 Retrieval Quality Benchmark"])

    with tab1:
        st.markdown("### Ask Knowledge Base Questions")
        st.info("The RAG system performs similarity search over `knowledge_base/` markdown files (strategies, metrics, biases, architecture).")

        query = st.text_input("Enter your quantitative research question:", value="What is look-ahead bias in backtesting?")

        if st.button("🔎 Submit RAG Query", type="primary"):
            with st.spinner("Searching FAISS vector index & generating grounded answer..."):
                rag_chain = RAGChain()
                res = rag_chain.answer_question(query)

                st.markdown("#### Grounded Answer")
                st.markdown(res["answer"])

                st.markdown("#### Referenced Sources")
                for src in res["sources"]:
                    st.markdown(f"- 📄 `{src}`")

                with st.expander("Retrieved Document Chunks (FAISS Top-K)", expanded=False):
                    for idx, chunk in enumerate(res["retrieved_chunks"]):
                        st.markdown(f"**Chunk #{idx+1}** (`{chunk['source']}`)")
                        st.code(chunk["text"], language="markdown")

    with tab2:
        st.markdown("### Empirical Retrieval Accuracy Benchmarking")
        st.write("Measures Top-1 and Top-3 accuracy across ground-truth evaluation question dataset.")

        if st.button("🧪 Run Retrieval Accuracy Test"):
            with st.spinner("Evaluating FAISS retrieval performance..."):
                eval_res = evaluate_rag_retrieval()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Test Questions", eval_res["total_questions"])
                col2.metric("Top-1 Accuracy", f"{eval_res['top_1_accuracy']:.1f}%")
                col3.metric("Top-3 Accuracy", f"{eval_res['top_3_accuracy']:.1f}%")

                st.markdown("#### Test Question Breakdown")
                st.table(eval_res["detailed_results"])
