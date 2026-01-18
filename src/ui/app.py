#!/usr/bin/env python3
"""Streamlit app for RAG chatbot."""
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.build_and_save_vectorstore import build_or_load_vectorstore
from rag.chain import create_rag_chain
from rag.retriever import create_retriever


# Page config
st.set_page_config(
    page_title="YSDA Admissions Assistant",
    page_icon="🎓",
    layout="wide",
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def initialize_rag():
    """Initialize RAG components."""
    if st.session_state.initialized:
        return

    api_key = ""
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY environment variable not set!")
        st.stop()

    with st.spinner("🔄 Initializing RAG system..."):
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small",
            base_url="https://api.vsellm.ru/",
        )

        # Build or load vectorstore
        chunks_paths = [
            "/Users/shchsergey/programming/herbarus-ysda-rag/src/splitting/data/knowledge_base_chunks.jsonl",
        ]
        index_path = "src/rag/data/vectorstore"

        vectorstore, was_loaded = build_or_load_vectorstore(
            chunks_paths=chunks_paths,
            index_path=index_path,
            embeddings=embeddings,
        )

        st.session_state.vectorstore = vectorstore

        # Create retriever
        retriever = create_retriever(vectorstore, search_type="similarity", k=5)

        # Create LLM
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            base_url="https://api.vsellm.ru/",
            temperature=0,
            timeout=60,
        )

        # Create RAG chain
        chain = create_rag_chain(retriever, llm)
        st.session_state.chain = chain

        st.session_state.initialized = True

        if was_loaded:
            st.success("✅ RAG system loaded from cache!")
        else:
            st.success("✅ RAG system initialized!")


# Initialize on first run
initialize_rag()

# UI
st.title("🎓 YSDA Admissions Assistant")
st.markdown("Задайте вопрос о поступлении в Школу анализа данных")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Введите ваш вопрос..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Думаю..."):
            try:
                response = st.session_state.chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Ошибка: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
