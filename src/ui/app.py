#!/usr/bin/env python3
"""Streamlit app for RAG chatbot (Agent only)."""
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.agent import build_agent
from rag.build_and_save_vectorstore import build_or_load_vectorstore

ESCALATIONS_PATH = (Path(__file__).resolve().parents[1] / "rag" / "data" / "escalations.csv")

st.set_page_config(
    page_title="YSDA Admissions Assistant",
    page_icon="🎓",
    layout="wide",
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(id(st.session_state))


def initialize_rag():
    """Initialize RAG components."""
    if st.session_state.initialized:
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY environment variable not set!")
        st.stop()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.vsellm.ru/")
    embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    with st.spinner("🔄 Initializing RAG system..."):
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model,
            base_url=base_url
        )

        chunks_paths = [
            "src/splitting/data/database.jsonl",
        ]
        index_path = "src/rag/data/vectorstore"

        vectorstore, was_loaded = build_or_load_vectorstore(
            chunks_paths=chunks_paths,
            index_path=index_path,
            embeddings=embeddings,
        )
        st.session_state.vectorstore = vectorstore

        llm = ChatOpenAI(
            api_key=api_key,
            model=llm_model,
            base_url=base_url,
            temperature=0,
            timeout=60,
        )

        agent = build_agent(
            llm=llm,
            vectorstore=vectorstore,
            k=6,
            user_id=st.session_state.session_id,
            escalation_log_path=ESCALATIONS_PATH,  # absolute
        )
        st.session_state.agent = agent

        st.session_state.initialized = True

        if was_loaded:
            st.success("✅ RAG system loaded from cache!")
        else:
            st.success("✅ RAG system initialized!")


initialize_rag()

# UI
st.title("🎓 YSDA Admissions Assistant")

tab_chat, tab_admin = st.tabs(["💬 Чат", "⚙️ Админ панель"])

with tab_chat:
    col_title, col_clear = st.columns([3, 1])
    with col_title:
        st.markdown("### Чат-бот для абитуриентов ШАД")
        st.markdown("Агент с обогащением запросов, проверкой актуальности и эскалацией")
    with col_clear:
        if st.button("🗑️ Очистить историю", key="clear_agent"):
            st.session_state.agent_messages = []
            st.rerun()

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Введите ваш вопрос...", key="agent_input"):
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Думаю..."):
                try:
                    chat_history = []
                    for msg in st.session_state.agent_messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))

                    response = st.session_state.agent(prompt, chat_history)

                    if isinstance(response, str) and response.lstrip().upper().startswith("ESCALATE"):
                        reason = response.split(":", 1)[-1].strip() if ":" in response else "Не могу ответить уверенно."
                        response = (
                            "Я не могу ответить уверенно по базе знаний ШАД.\n\n"
                            f"Причина: {reason}\n\n"
                            "Я записал вопрос для кураторов. Если уточнишь трек/год набора/этап — попробую ответить точнее."
                        )

                    st.markdown(response)
                    st.session_state.agent_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Ошибка: {str(e)}"
                    st.error(error_msg)
                    st.session_state.agent_messages.append({"role": "assistant", "content": error_msg})

with tab_admin:
    st.markdown("### Админ панель - Неотвеченные вопросы")

    escalation_file = ESCALATIONS_PATH

    if escalation_file.exists():
        try:
            df = pd.read_csv(escalation_file)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            st.markdown(f"**Всего эскалаций:** {len(df)}")

            col1, col2 = st.columns(2)
            with col1:
                search_query = st.text_input("🔍 Поиск по вопросу", key="search_question")
            with col2:
                date_range = None
                if "timestamp" in df.columns and len(df) > 0:
                    try:
                        valid_ts = df["timestamp"].dropna()
                        if len(valid_ts) > 0:
                            date_range = st.date_input(
                                "📅 Фильтр по дате",
                                value=(valid_ts.min().date(), valid_ts.max().date()),
                                key="date_filter"
                            )
                    except Exception:
                        date_range = None

            filtered_df = df.copy()
            if search_query and "question" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["question"].astype(str).str.contains(search_query, case=False, na=False)
                ]

            if "timestamp" in filtered_df.columns and date_range is not None:
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df["timestamp"].dt.date >= date_range[0]) &
                        (filtered_df["timestamp"].dt.date <= date_range[1])
                    ]

            st.markdown(f"**Найдено:** {len(filtered_df)}")

            if len(filtered_df) > 0:
                display_df = filtered_df.copy()
                if "timestamp" in display_df.columns:
                    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать CSV",
                    data=csv_data,
                    file_name=f"escalations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Нет данных, соответствующих фильтрам")

        except Exception as e:
            st.error(f"Ошибка при чтении файла: {str(e)}")
    else:
        st.info("Файл с эскалациями пока не создан. Эскалации будут появляться здесь после первых неотвеченных вопросов.")
