"""
memory_page.py

Streamlit page for Conversation Memory Demonstration & State Reset.
(29-Aug Milestone)
"""

import streamlit as st
from rag.memory import ConversationMemory
from rag.rag_chain import RAGChain


def render_memory_page():
    st.markdown("## 💬 Multi-Turn Conversation Memory Demo")
    st.caption("Demonstrating 29-Aug-2026 Milestone: Conversational State Memory with [Clear Memory]")

    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = ConversationMemory()

    memory: ConversationMemory = st.session_state["chat_memory"]

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🧹 [Clear Memory]", use_container_width=True):
            memory.clear_memory()
            st.session_state["chat_messages"] = []
            st.success("Conversation memory cleared!")
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Display chat history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a follow-up question (e.g. 'Explain SMA.', then 'What are its limitations?')...")
    if user_input:
        # Append user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Resolve context query via memory
        contextual_query = memory.get_contextual_query(user_input)
        memory.add_user_message(user_input)

        # Answer via RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Processing with memory context..."):
                rag_chain = RAGChain()
                res = rag_chain.answer_question(contextual_query)
                answer = res["answer"]
                st.markdown(answer)
                st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                memory.add_assistant_message(answer)
