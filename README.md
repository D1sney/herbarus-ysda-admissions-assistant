# YSDA Admissions Assistant

RAG (Retrieval-Augmented Generation) система для ассистента по поступлению в Школу анализа данных (ШАД) от Яндекса.

## Описание

Система использует RAG для ответов на вопросы абитуриентов о поступлении в ШАД. База знаний собирается из:
- Официального сайта ШАД (`shad.yandex.ru`)
- Telegram-канала абитуриентов (`@vse_v_shad`)

## Стек разработки

### Язык и окружение
- **Python 3.10+** - основной язык разработки

### RAG и LLM
- **LangChain** - фреймворк для построения RAG-приложений
  - `langchain-core` - базовые компоненты
  - `langchain-community` - интеграции с внешними сервисами
  - `langchain-openai` - интеграция с OpenAI API
  - `langchain-text-splitters` - разбивка документов на чанки
- **OpenAI API** - для LLM и embeddings (совместимый API, например `vsellm.ru`)
- **FAISS** - векторная база данных для семантического поиска

### Веб-скрапинг и обработка данных
- **requests** - HTTP-запросы для краулинга
- **BeautifulSoup4** - парсинг HTML
- **pandas** - обработка структурированных данных

### Интерфейсы
- **Streamlit** - веб-интерфейс для чата и админ-панели
- **python-telegram-bot** - Telegram бот

### Дополнительные библиотеки
- **pydantic** - валидация данных и схемы
- **httpx** - HTTP-клиент для асинхронных запросов

## Архитектура

```
┌─────────────────┐
│  Data Sources   │
│  (Web + TG)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion      │
│  (Crawl/Parse)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Splitting      │
│  (Chunking)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│    (FAISS)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Agent      │
│  (Retrieval +   │
│   Generation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   UI Layer      │
│ (Streamlit/TG)  │
└─────────────────┘
```

## Установка

### Требования

- Python 3.10+
- API ключ для OpenAI-совместимого API (например, `vsellm.ru`)

### Шаги установки

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd herbarus-ysda-rag
```

2. Создайте виртуальное окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Настройте переменные окружения:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.vsellm.ru/"  # опционально
export EMBEDDING_MODEL="openai/text-embedding-3-large"  # опционально
export LLM_MODEL="gpt-4o-mini"  # опционально
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"  # только для Telegram бота
```

## Использование

### 1. Сбор базы знаний

#### Веб-сайт ШАД

```bash
python3 src/ingest/web/build_knowledge_base.py \
    --start-url https://shad.yandex.ru/ \
    --max-depth 1 \
    --delay 0.5 \
    --min-chars 200
```

Параметры:
- `--start-url` - стартовая URL (можно указать несколько раз)
- `--max-depth` - максимальная глубина обхода (по умолчанию: 1)
- `--delay` - задержка между запросами в секундах (по умолчанию: 0.5)
- `--min-chars` - минимальная длина текста страницы (по умолчанию: 200)
- `--data-dir` - директория для данных (по умолчанию: `src/ingest/web/data`)
- `--no-clear` - не очищать директорию перед запуском

Результат: `src/ingest/web/data/processed/text/corpus.jsonl`

#### Telegram-канал

1. Экспортируйте данные из Telegram в `result.json`
2. Парсинг:
```bash
python3 src/ingest/telegram/parse_telegram_export.py
```
3. Оценка релевантности (опционально):
```bash
python3 src/ingest/telegram/evaluate_relevance_langchain.py
```
4. Обогащение базы знаний:
```bash
python3 src/ingest/telegram/enrich_tg_knowledge_base.py
```

Результат: `src/ingest/telegram/tg_knowledge_base_boosted.jsonl`

### 2. Разбивка документов на чанки

```bash
python3 src/splitting/split_corpus.py \
    --input-jsonl src/ingest/web/data/processed/text/corpus.jsonl \
    --input-jsonl src/ingest/telegram/tg_knowledge_base_boosted.jsonl \
    --output-jsonl src/splitting/data/knowledge_base_chunks.jsonl \
    --max-chars 1000 \
    --overlap 200
```

Параметры:
- `--input-jsonl` - входные JSONL файлы (можно указать несколько раз)
- `--output-jsonl` - выходной JSONL файл (по умолчанию: `src/splitting/data/chunks.jsonl`)
- `--max-chars` - максимальный размер чанка (по умолчанию: 1000)
- `--overlap` - перекрытие между чанками (по умолчанию: 200)

### 3. Построение векторной базы данных

Векторная БД создается автоматически при первом запуске UI или Telegram бота.

Или вручную:
```python
from rag.build_and_save_vectorstore import build_or_load_vectorstore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.vsellm.ru/")
)

vectorstore, was_loaded = build_or_load_vectorstore(
    chunks_paths=["src/splitting/data/knowledge_base_chunks.jsonl"],
    index_path="src/rag/data/vectorstore",
    embeddings=embeddings,
)
```

### 4. Запуск интерфейсов

#### Streamlit Web UI

```bash
streamlit run src/ui/app.py
```

Интерфейс включает:
- **Чат** - общение с RAG агентом
- **Админ панель** - просмотр неотвеченных вопросов (эскалаций)

#### Telegram Bot

```bash
python3 src/ui/telegram_bot.py
```

Команды бота:
- `/start` - приветствие
- `/reset` - очистить историю чата

## Структура проекта

```
herbarus-ysda-rag/
├── src/
│   ├── ingest/              # Сбор базы знаний
│   │   ├── web/             # Парсинг сайта shad.yandex.ru
│   │   │   ├── crawl_shad.py           # Краулинг сайта
│   │   │   ├── extract_text.py         # Извлечение текста из HTML
│   │   │   ├── build_corpus.py         # Построение корпуса
│   │   │   └── build_knowledge_base.py # Главный скрипт
│   │   └── telegram/         # Парсинг Telegram-канала
│   │       ├── parse_telegram_export.py
│   │       ├── evaluate_relevance_langchain.py
│   │       └── enrich_tg_knowledge_base.py
│   ├── splitting/            # Разбивка документов
│   │   ├── split_corpus.py  # Разбивка на чанки
│   │   └── load_documents.py # Загрузка документов
│   ├── rag/                  # RAG компоненты
│   │   ├── vectorstore.py    # Управление FAISS
│   │   ├── retriever.py     # Ретривер
│   │   ├── chain.py          # Простая RAG цепочка
│   │   ├── agent.py          # Продвинутый RAG агент
│   │   └── build_and_save_vectorstore.py
│   └── ui/                   # Интерфейсы
│       ├── app.py            # Streamlit приложение
│       └── telegram_bot.py   # Telegram бот
├── requirements.txt
└── README.md
```

## Компоненты системы

### RAG Agent

Продвинутый агент с:
- **Обогащением запросов** - исправление опечаток, уточнение местоимений
- **Проверкой актуальности** - для time-sensitive вопросов
- **Автоматической эскалацией** - логирование неотвеченных вопросов
- **Историей чата** - контекст предыдущих сообщений

### Эскалация

Когда агент не может ответить уверенно, вопрос логируется в CSV:
- `src/rag/data/escalations.csv` - для админ панели
- `src/rag/data/escalations_debug.jsonl` - расширенная отладочная информация

Формат CSV:
- `user_id` - идентификатор пользователя
- `timestamp` - время эскалации
- `question` - вопрос пользователя
- `reason` - причина эскалации

### Промпты

Системный промпт находится в `src/rag/chain.py` и включает:
- Инструкции для ассистента
- Правила работы с контекстом
- Примеры правильных ответов
- Требования к формату ответов

## Переменные окружения

| Переменная | Описание | Обязательная | По умолчанию |
|------------|----------|--------------|--------------|
| `OPENAI_API_KEY` | API ключ для LLM и embeddings | Да | - |
| `OPENAI_BASE_URL` | Базовый URL API | Нет | `https://api.vsellm.ru/` |
| `EMBEDDING_MODEL` | Модель для embeddings | Нет | `openai/text-embedding-3-large` |
| `LLM_MODEL` | Модель для генерации | Нет | `gpt-4o-mini` |
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота | Только для бота | - |

## Особенности

- **Автоматическое кэширование** векторной БД - не пересоздается при каждом запуске
- **Логирование эскалаций** - все неотвеченные вопросы сохраняются для анализа
- **Поддержка истории чата** - агент учитывает контекст предыдущих сообщений
- **Проверка актуальности** - для вопросов с датами проверяется свежесть данных
- **Обогащение запросов** - автоматическое улучшение вопросов пользователя

## Разработка

### Формат данных

Все данные хранятся в JSONL формате:

```json
{
  "text": "Содержимое документа...",
  "metadata": {
    "url": "https://...",
    "date": "2025-01-20T10:30:00",
    "source": "web|telegram"
  }
}
```

### Добавление новых источников

1. Создайте скрипт парсинга в `src/ingest/`
2. Сохраните результат в JSONL формате
3. Добавьте путь в `split_corpus.py` при разбивке
4. Пересоздайте векторную БД
