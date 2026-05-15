import streamlit as st
import os
from rag_engine import get_answer, build_vectorstore

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

st.set_page_config(page_title="Study Buddy RAG", page_icon="📚", layout="centered")

with st.sidebar:
    st.title("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Processing your document..."):
            vs = build_vectorstore(uploaded_file)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"Ready: {uploaded_file.name}")
            else:
                st.error("Failed to process PDF.")
    if st.session_state.doc_name:
        st.info(f"Active: {st.session_state.doc_name}")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Upload any PDF\n2. Ask questions\n3. Get answers with sources")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.title("📚 Study Buddy")
st.caption("Ask questions from your uploaded PDF notes or textbook")

if not st.session_state.vectorstore:
    st.info("Please upload a PDF from the sidebar to start.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            if entry["refused"]:
                st.warning(entry["answer"])
            else:
                st.write(entry["answer"])
                if entry.get("sources"):
                    with st.expander("View Source Chunks"):
                        for i, src in enumerate(entry["sources"], 1):
                            st.info(f"Chunk {i}: {src[:300]}...")

    query = st.chat_input("Ask a question from your document...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_answer(query, st.session_state.vectorstore)
            if result["refused"]:
                st.warning(result["answer"])
            else:
                st.write(result["answer"])
                if result.get("sources"):
                    with st.expander("View Source Chunks"):
                        for i, src in enumerate(result["sources"], 1):
                            st.info(f"Chunk {i}: {src[:300]}...")
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "refused": result["refused"]
        })
