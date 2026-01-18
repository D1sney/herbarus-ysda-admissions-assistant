from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


# -------------------------
# Heuristics
# -------------------------

TIME_SENSITIVE_PATTERNS = [
    r"\bсрок(и|ов)?\b", r"\bдедлайн\b", r"\bкогда\b", r"\bсейчас\b", r"\bв этом году\b",
    r"\bнабор\b", r"\bолимпиад(а|ы)\b", r"\bэкзамен\b", r"\bтест(ирование)?\b",
    r"\b202[4-9]\b",
]

def is_time_sensitive(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(p, q) for p in TIME_SENSITIVE_PATTERNS)

def parse_iso_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


# -------------------------
# Prompt: enrich query
# -------------------------

ENRICH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты помогаешь улучшить запрос пользователя к базе знаний ШАД.\n"
     "Сделай:\n"
     "- исправь опечатки/ошибки\n"
     "- убери лишнее\n"
     "- если можно, уточни местоимения по контексту чата\n"
     "Верни JSON строго такого вида:\n"
     "{{\n"
     "  \"clean_query\": \"...\",\n"
     "  \"intent\": \"поступление|учеба|курсы|наука|тьюторы|прочее\",\n"
     "  \"needs_clarification\": true|false,\n"
     "  \"clarifying_question\": \"...\"\n"
     "}}\n"
     "Если уточнение не нужно — needs_clarification=false и clarifying_question=\"\"."
    ),
    MessagesPlaceholder("chat_history"),
    ("user", "{question}"),
])

# -------------------------
# Prompt: answer using retrieved docs
# -------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — ассистент по поступлению в ШАД.\n"
     "Отвечай ТОЛЬКО на основе контекста.\n"
     "Если в контексте нет ответа — напиши: \"В базе знаний нет точного ответа\" и предложи уточнить вопрос.\n"
     "Не придумывай.\n"
     "В конце добавь блок \"Источники:\" и перечисли URL (и source/date если есть) для использованных фрагментов.\n"
    ),
    ("user",
     "Вопрос: {question}\n\n"
     "Контекст:\n{context}\n\n"
     "Ответ:"
    ),
])

def format_docs_with_meta(docs: List[Document], max_docs: int = 6) -> str:
    lines = []
    for i, d in enumerate(docs[:max_docs]):
        md = d.metadata or {}
        url = md.get("url", "")
        source = md.get("source", "")
        date = md.get("date", "")
        header = f"[Документ {i+1} | source={source} | date={date} | url={url}]"
        lines.append(header + "\n" + (d.page_content or ""))
    return "\n\n".join(lines).strip()


def escalate(reason: str) -> str:
    return (
        "Я не могу ответить уверенно по базе знаний ШАД.\n\n"
        f"Причина: {reason}\n\n"
        "Что можно сделать:\n"
        "- уточнить вопрос (город/трек/год набора/этап)\n"
        "- проверить актуальную информацию на сайте ШАД\n"
        "- задать вопрос куратору/в чате абитуриентов\n"
    )


# -------------------------
# Public: build_agent (returns callable)
# -------------------------

def build_agent(
    llm: ChatOpenAI,
    vectorstore: FAISS,
    k: int = 6,
    stale_days_threshold: int = 180,
):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def run(question: str, chat_history: List[BaseMessage]) -> str:
        # 1) enrich
        enrich_msgs = ENRICH_PROMPT.format_messages(chat_history=chat_history, question=question)
        enrich_out = llm.invoke(enrich_msgs).content

        try:
            enrich = json.loads(enrich_out)
        except Exception:
            enrich = {
                "clean_query": question.strip(),
                "intent": "прочее",
                "needs_clarification": False,
                "clarifying_question": "",
            }

        if enrich.get("needs_clarification") and enrich.get("clarifying_question"):
            return enrich["clarifying_question"]

        clean_query = enrich.get("clean_query", question).strip()

        # 2) retrieve
        docs = retriever.invoke(clean_query)

        if not docs:
            return escalate("Не нашёл релевантных документов по запросу.")

        # 3) stale check (для time-sensitive)
        if is_time_sensitive(clean_query):
            dates = [parse_iso_dt((d.metadata or {}).get("date", "")) for d in docs]
            dates = [d for d in dates if d is not None]
            if dates:
                newest = max(dates)
                if newest < (datetime.utcnow() - timedelta(days=stale_days_threshold)):
                    return escalate(
                        f"Вопрос зависит от актуальных дат, а найденные документы старые (новейший: {newest.isoformat()})."
                    )

        # 4) answer
        context = format_docs_with_meta(docs, max_docs=k)
        ans_msgs = ANSWER_PROMPT.format_messages(question=question, context=context)
        answer = llm.invoke(ans_msgs).content.strip()

        # дополнительная защита: если модель сказала, что нет ответа — эскалируем
        if "В базе знаний нет точного ответа" in answer:
            return escalate("В найденных документах нет точного ответа.")

        return answer

    return run
