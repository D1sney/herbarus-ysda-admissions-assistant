from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from rag.chain import get_default_prompt_template

ESCALATIONS_DEFAULT_PATH = (Path(__file__).resolve().parent / "data" / "escalations.csv")
ESCALATIONS_DEBUG_JSONL_PATH = (Path(__file__).resolve().parent / "data" / "escalations_debug.jsonl")
_ESCALATE_RE = re.compile(r"^\s*ESCALATE\s*[:：]\s*(.*)$", re.IGNORECASE)

TIME_SENSITIVE_PATTERNS = [
    r"\bсрок(и|ов)?\b", r"\bдедлайн\b", r"\bкогда\b", r"\bсейчас\b", r"\bв этом году\b",
    r"\bнабор\b", r"\bолимпиад(а|ы)\b", r"\bэкзамен\b", r"\bтест(ирование)?\b",
    r"\b202[4-9]\b",
]

UNSURE_PATTERNS = [
    r"\bне знаю\b",
    r"\bне уверен\b",
    r"\bнет точного ответа\b",
    r"\bв базе знаний нет\b",
    r"\bв контексте нет\b",
    r"\bне могу ответить уверенно\b",
    r"\bне наш(ё|е)л\b",
]

def is_time_sensitive(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(p, q) for p in TIME_SENSITIVE_PATTERNS)

def looks_unsure(answer: str) -> bool:
    a = (answer or "").lower()
    return any(re.search(p, a) for p in UNSURE_PATTERNS)

def parse_iso_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


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

def get_answer_prompt() -> ChatPromptTemplate:
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = get_default_prompt_template(current_date)
    return ChatPromptTemplate.from_template(system_prompt)

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

def docs_meta_summary(docs: List[Document], max_docs: int = 6) -> Dict[str, Any]:
    urls = []
    dates = []
    for d in docs[:max_docs]:
        md = d.metadata or {}
        u = md.get("url")
        if u:
            urls.append(u)
        dt = md.get("date")
        if dt:
            dates.append(dt)
    return {"top_urls": urls, "doc_dates": dates}

def chat_tail_as_json(chat_history: List[BaseMessage], n: int = 6) -> str:
    tail = chat_history[-n:] if chat_history else []
    out = []
    for m in tail:
        role = getattr(m, "type", None) or m.__class__.__name__
        content = getattr(m, "content", "")
        out.append({"role": role, "content": content})
    return json.dumps(out, ensure_ascii=False)

def log_escalation(
    user_id: str,
    question: str,
    reason: str,
    escalation_log_path: str | Path = ESCALATIONS_DEFAULT_PATH,
) -> None:
    log_path = Path(escalation_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "timestamp", "question", "reason"])
        writer.writerow([user_id, datetime.now().isoformat(), question, reason])


def log_escalation_debug_jsonl(
    user_id: str,
    question: str,
    reason: str,
    clean_query: str = "",
    intent: str = "",
    clarifying_question: str = "",
    docs: Optional[List[Document]] = None,
    chat_history: Optional[List[BaseMessage]] = None,
    debug_log_path: str | Path = ESCALATIONS_DEBUG_JSONL_PATH,
) -> None:
    docs = docs or []
    chat_history = chat_history or []
    meta = docs_meta_summary(docs, max_docs=6)

    payload = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "reason": reason,
        "clean_query": clean_query,
        "intent": intent,
        "clarifying_question": clarifying_question,
        "top_urls": meta["top_urls"],
        "doc_dates": meta["doc_dates"],
        "retrieved_preview": format_docs_with_meta(docs, max_docs=5),
        "chat_tail": json.loads(chat_tail_as_json(chat_history, n=6)) if chat_history else [],
    }

    path = Path(debug_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def escalate_to_human(
    reason: str,
    user_id: str | None = None,
    question: str | None = None,
    clean_query: str = "",
    intent: str = "",
    clarifying_question: str = "",
    docs: Optional[List[Document]] = None,
    chat_history: Optional[List[BaseMessage]] = None,
    escalation_log_path: str | Path = ESCALATIONS_DEFAULT_PATH,
) -> str:
    if user_id is not None and question is not None:
        try:
            log_escalation(
                user_id=user_id,
                question=question,
                reason=reason,
                escalation_log_path=escalation_log_path,
            )
        except Exception as e:
            print(f"[escalate_to_human] Failed to log escalation(CSV): {e}")

        try:
            log_escalation_debug_jsonl(
                user_id=user_id,
                question=question,
                reason=reason,
                clean_query=clean_query,
                intent=intent,
                clarifying_question=clarifying_question,
                docs=docs,
                chat_history=chat_history,
            )
        except Exception as e:
            print(f"[escalate_to_human] Failed to log escalation(JSONL): {e}")

    msg = (
        "Я не могу ответить уверенно по базе знаний ШАД.\n\n"
        f"Причина: {reason}\n\n"
        "Я записал вопрос для кураторов. Если уточнишь трек/год набора/этап — попробую ответить точнее."
    )
    if clarifying_question:
        msg += "\n\nЧтобы ускорить ответ, уточни, пожалуйста:\n" + clarifying_question
    return msg

def build_agent(
    llm: ChatOpenAI,
    vectorstore: FAISS,
    k: int = 6,
    stale_days_threshold: int = 180,
    user_id: str | None = None,
    escalation_log_path: str | Path = ESCALATIONS_DEFAULT_PATH,
):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    def _strip_code_fence(s: str) -> str:
        s = (s or "").strip()
        if not s.startswith("```"):
            return s
        s = s.strip("`").strip()
        if "\n" in s:
            first, rest = s.split("\n", 1)
            if first.strip().lower() in {"json", "javascript", "js"}:
                s = rest.strip()
        if "\n" in s and s.splitlines()[-1].strip() == "":
            s = s.strip()
        return s

    def run(question: str, chat_history: List[BaseMessage], user_id_override: str | None = None) -> str:
        effective_user_id = user_id_override if user_id_override is not None else user_id

        enrich_msgs = ENRICH_PROMPT.format_messages(chat_history=chat_history, question=question)
        enrich_out = llm.invoke(enrich_msgs).content
        enrich_out = _strip_code_fence(enrich_out)

        try:
            enrich = json.loads(enrich_out)
        except Exception:
            enrich = {
                "clean_query": question.strip(),
                "intent": "прочее",
                "needs_clarification": False,
                "clarifying_question": "",
            }

        clean_query = (enrich.get("clean_query") or question).strip()
        intent = (enrich.get("intent") or "прочее").strip()
        needs_clar = bool(enrich.get("needs_clarification"))
        clar_q = (enrich.get("clarifying_question") or "").strip()

        # Retry логика для нестабильных embeddings (OpenRouter и др.)
        docs = None
        last_error = None
        for attempt in range(3):
            try:
                docs = retriever.invoke(clean_query)
                if docs is not None:
                    break
            except Exception as e:
                last_error = e
                import time
                time.sleep(0.5 * (attempt + 1))

        if docs is None and last_error:
            raise last_error

        if not docs:
            return escalate_to_human(
                "Не нашёл релевантных документов по запросу.",
                user_id=effective_user_id,
                question=question,
                clean_query=clean_query,
                intent=intent,
                clarifying_question=clar_q if needs_clar else "",
                docs=[],
                chat_history=chat_history,
                escalation_log_path=escalation_log_path,
            )

        if is_time_sensitive(clean_query):
            dates = [parse_iso_dt((d.metadata or {}).get("date", "")) for d in docs]
            dates = [d for d in dates if d is not None]
            if dates:
                newest = max(dates)
                if newest < (datetime.utcnow() - timedelta(days=stale_days_threshold)):
                    return escalate_to_human(
                        f"Вопрос зависит от актуальных дат, а найденные документы старые (новейший: {newest.isoformat()}).",
                        user_id=effective_user_id,
                        question=question,
                        clean_query=clean_query,
                        intent=intent,
                        clarifying_question=clar_q if needs_clar else "",
                        docs=docs,
                        chat_history=chat_history,
                        escalation_log_path=escalation_log_path,
                    )

        context = format_docs_with_meta(docs, max_docs=10)
        answer_prompt = get_answer_prompt()
        ans_msgs = answer_prompt.format_messages(question=question, context=context)
        answer = (llm.invoke(ans_msgs).content or "").strip()

        m = _ESCALATE_RE.match(answer or "")
        if m:
            reason = (m.group(1) or "").strip() or "LLM определил, что не может дать уверенный ответ."
            return escalate_to_human(
                reason,
                user_id=effective_user_id,
                question=question,
                clean_query=clean_query,
                intent=intent,
                clarifying_question=clar_q if needs_clar else "",
                docs=docs,
                chat_history=chat_history,
                escalation_log_path=escalation_log_path,
            )

        if looks_unsure(answer):
            if needs_clar and clar_q:
                return clar_q

            return escalate_to_human(
                "В контексте нет информации по этому вопросу.",
                user_id=effective_user_id,
                question=question,
                clean_query=clean_query,
                intent=intent,
                clarifying_question="",
                docs=docs,
                chat_history=chat_history,
                escalation_log_path=escalation_log_path,
            )

        return answer

    return run
