import os
import json
import re
import time
from pathlib import Path
from typing import List, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

INPUT_JSONL = Path("src/ingest/telegram/tg_knowledge_base.jsonl")
OUTPUT_JSONL = Path("src/ingest/telegram/tg_knowledge_base_boosted.jsonl")


llm_http_client = httpx.Client(timeout=60.0, verify=False)

llm = ChatOpenAI(
    api_key="sk-pmozajQBrdWVKqeAYs4n8A",
    model="gpt-4o-mini",
    base_url="https://api.vsellm.ru/",
    temperature=0
    )

MAX_RETRIES = 3
SLEEP_ON_RETRY_SEC = 2.0

_EMOJI_RE = re.compile(
    "["

    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\u200d"
    "\uFE0F"
    "]",
    flags=re.UNICODE,
)

_BULLETS_RE = re.compile(r"[•●◦▪️▫️■□◆◇▶►✓✔✗✖✘➤➔→⇒]+", re.UNICODE)
_WS_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")

def clean_telegram_text(text: str) -> str:
    t = text or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _EMOJI_RE.sub("", t)
    t = _BULLETS_RE.sub("", t)
    t = _WS_RE.sub(" ", t)
    t = _MANY_NEWLINES_RE.sub("\n\n", t)
    return t.strip()

ContentType = Literal["поступление", "дедлайн", "ответ_на_вопрос", "объявление", "шум"]

class ClassifyOut(BaseModel):
    тип_контента: ContentType = Field(
        description="Один из: поступление, дедлайн, ответ_на_вопрос, объявление, шум"
    )
    причина: str = Field(
        description="Короткое объяснение выбора типа (1 фраза)."
    )

class EnrichOut(BaseModel):
    summary: str = Field(description="1-2 предложения, кратко о сути.")
    tags: List[str] = Field(description="Список тегов на русском, 3-8 штук.")
    keywords: List[str] = Field(description="Список ключевых фраз, 5-12 штук.")
    possible_questions: List[str] = Field(description="Список возможных вопросов пользователя, 3-5 штук.")
    content: str = Field(description="Нормализованный текст без эмодзи, без искажений смысла.")

classify_parser = PydanticOutputParser(pydantic_object=ClassifyOut)
enrich_parser = PydanticOutputParser(pydantic_object=EnrichOut)

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты помощник, который классифицирует Telegram-сообщения абитуриентского канала ШАД. "
     "Отвечай строго в формате, который задаёт parser.\n\n{format_instructions}\n\n"
     "Правила:\n"
     "- 'поступление' — требования, этапы отбора, треки, документы, условия поступления.\n"
     "- 'дедлайн' — даты/сроки/таймлайны, расписания этапов, когда что заканчивается.\n"
     "- 'ответ_на_вопрос' — сообщение явно отвечает на вопрос/разъясняет конкретную ситуацию.\n"
     "- 'объявление' — анонсы, новости, запись трансляции, запуск форм, организационные объявления.\n"
     "- 'шум' — приветствия, флуд, оффтоп, сообщения без полезной инфы для поступления/учёбы.\n"
     "Если сомневаешься между 'поступление' и 'объявление':\n"
     "- если есть действие/новость/анонс — 'объявление'\n"
     "- если это справочная информация — 'поступление'."
    ),
    ("user", "Текст:\n{clean_text}")
])

ENRICH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты преобразуешь Telegram-сообщение про ШАД в структурированный фрагмент базы знаний.\n"
     "Требования:\n"
     "- Убери эмодзи, пиктограммы, лишние маркеры, но сохрани смысл.\n"
     "- Не добавляй фактов, которых нет в тексте.\n"
     "- Пиши по-русски.\n"
     "- Теги — на русском, в snake_case или с подчёркиваниями.\n"
     "- Возможные вопросы — как от абитуриента (3-5), без воды.\n\n"
     "{format_instructions}"
    ),
    ("user",
     "TYPE: {content_type}\n"
     "URL: {url}\n"
     "DATE: {date}\n\n"
     "Исходный текст (уже очищенный):\n{clean_text}"
    )
])

classify_chain = (
    CLASSIFY_PROMPT.partial(format_instructions=classify_parser.get_format_instructions())
    | llm
    | classify_parser
)

enrich_chain = (
    ENRICH_PROMPT.partial(format_instructions=enrich_parser.get_format_instructions())
    | llm
    | enrich_parser
)

def format_enriched_chunk(source_url: str, out: EnrichOut) -> str:
    tags = ", ".join(out.tags)
    keywords = "; ".join(out.keywords)

    questions = "\n".join([f"- {q}" for q in out.possible_questions])

    return (
        "--- НАЧАЛО ЧАНКА ---\n\n"
        "SOURCE:\n"
        f"{source_url}\n\n"
        "SUMMARY:\n"
        f"{out.summary}\n\n"
        "TAGS:\n"
        f"{tags}\n\n"
        "KEYWORDS:\n"
        f"{keywords}\n\n"
        "ВОЗМОЖНЫЕ ВОПРОСЫ:\n"
        f"{questions}\n\n"
        "CONTENT:\n"
        f"{out.content}\n"
    )


def call_with_retries(fn, *args, **kwargs):
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_ON_RETRY_SEC * attempt)
            else:
                raise last_err

def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_JSONL}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped_noise = 0
    written = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = (item.get("text") or "").strip()
            metadata = item.get("metadata") or {}

            url = str(metadata.get("url") or "").strip()
            date = str(metadata.get("date") or "").strip()

            clean_text = clean_telegram_text(text)

            if len(clean_text) < 20:
                skipped_noise += 1
                continue

            cls: ClassifyOut = call_with_retries(
                classify_chain.invoke,
                {"clean_text": clean_text}
            )

            if cls.тип_контента == "шум":
                skipped_noise += 1
                continue

            enr: EnrichOut = call_with_retries(
                enrich_chain.invoke,
                {
                    "content_type": cls.тип_контента,
                    "url": url,
                    "date": date,
                    "clean_text": clean_text,
                }
            )

            enriched_text = format_enriched_chunk(url, enr)

            out_metadata = dict(metadata)
            out_metadata["source"] = "telegram"
            out_metadata["content_type"] = cls.тип_контента

            record = {"text": enriched_text, "metadata": out_metadata}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if total % 50 == 0:
                print(f"[progress] total={total} written={written} skipped_noise={skipped_noise}")

    print(f"DONE. total={total} written={written} skipped_noise={skipped_noise}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
