"""
# app.py
# Task: 4 - Interactive Chat Interface
# Created by: Addisu Taye
# Date: July 4, 2025
# Purpose: Build web interface for querying complaint data
# Key Features: Streamlit UI with question input and source display
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline

st.title("💬 Complaint Analyst Chatbot")

@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about complaints"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response, sources = rag.run(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("Sources"):
            for s in sources:
                st.markdown(s["text"][:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": response})