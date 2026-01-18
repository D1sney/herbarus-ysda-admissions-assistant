
#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.vectorstore import load_vectorstore
from rag.agent import build_agent

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN environment variable")

INDEX_PATH = Path("src/rag/data/vectorstore")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.vsellm.ru/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

history_store: Dict[int, Deque] = {}


def get_history(chat_id: int) -> List:
    if chat_id not in history_store:
        history_store[chat_id] = deque(maxlen=6)
    return list(history_store[chat_id])


def append_user(chat_id: int, text: str):
    from langchain_core.messages import HumanMessage
    if chat_id not in history_store:
        history_store[chat_id] = deque(maxlen=6)
    history_store[chat_id].append(HumanMessage(content=text))


def append_ai(chat_id: int, text: str):
    from langchain_core.messages import AIMessage
    if chat_id not in history_store:
        history_store[chat_id] = deque(maxlen=6)
    history_store[chat_id].append(AIMessage(content=text))

def build_runtime():
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
        base_url=OPENAI_BASE_URL,
    )

    vectorstore: FAISS = load_vectorstore(INDEX_PATH, embeddings)

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )

    run_agent = build_agent(llm=llm, vectorstore=vectorstore, k=6)
    return run_agent


RUN_AGENT = build_runtime()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я YSDA Admissions Assistant.\n"
        "Задай вопрос про ШАД: поступление, учёба, курсы, тьюторы."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    history_store.pop(chat_id, None)
    await update.message.reply_text("История очищена.")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()

    if not user_text:
        return

    append_user(chat_id, user_text)
    chat_history = get_history(chat_id)

    import asyncio
    try:
        answer = await asyncio.to_thread(RUN_AGENT, user_text, chat_history, str(chat_id))
    except Exception as e:
        answer = f"Ошибка: {e}"

    append_ai(chat_id, answer)
    await update.message.reply_text(answer)


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    app.run_polling()


if __name__ == "__main__":
    main()
