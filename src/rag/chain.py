import datetime
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single string.

    Args:
        docs: List of Document objects from retriever.

    Returns:
        Formatted string with document contents.
    """
    print(f"[chain] Formatting {len(docs)} documents...")
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"[chain] Total context size: {total_chars} characters")
    formatted = "\n\n".join([d.page_content for d in docs])
    print(f"[chain] Formatted context ready ({len(formatted)} chars)")
    return formatted


def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt_template: str | None = None,
) -> RunnablePassthrough:
    """Create a RAG chain from retriever and LLM.

    Args:
        retriever: Retriever instance.
        llm: Language model instance.
        prompt_template: Custom prompt template. If None, uses default.

    Returns:
        Runnable chain that takes a question string and returns an answer.
    """

    current_date = datetime.now().strftime("%Y-%m-%d")
    
    if prompt_template is None:
        prompt_template = """Ты — ассистент по поступлению в Школу анализа данных (ШАД) от Яндекса. Твоя задача — давать точные, фактические и полезные ответы на вопросы абитуриентов, основываясь **исключительно** на предоставленной тебе информации из базы знаний из поля Котекст, в котором идут контексты, осортированные по релевантности.

### Важные правила:
1. **Не выдумывай** информацию. Если в контексте нет ответа или он не релевантен вопросу вызови инструмент escalate.

2. **Ссылайся на источники**. В конце ответа обязательно добавляй ссылку на наиболее релевантный источник (поле `SOURCE` в каждом контексте), например:  
   > Источник: https://t.me/vse_v_shad/7

3. **Учитывай дату**. Используй поле `DATE` из каждого контекста для определения его актуальности.  
   - Сравнивай с текущей датой (сегодня: {current_date}).  
   - Если информация устарела — предупреди об этом.  
   - Пример:  
     > Эта информация актуальна на 2025 год. Проверь официальный сайт на предмет обновлений или задай вопрос к кураторам в чате @vse_v_shad.

4. **Если вопрос требует выбора или сравнения**, но в контексте нет всех вариантов — объясни, что ты ограничен доступными данными.

5. **Формат ответа**:  
   - Кратко, чётко, без лишних слов.  
   - Без эмоций, без "привет", без "друзья".  
   - Только по делу.

### Как работать с контекстом:
Каждый фрагмент контекста имеет структуру:
- `SOURCE`: ссылка на источник.
- `DATE`: дата публикации в формате ISO 8601 (YYYY-MM-DDTHH:MM:SS).
- `SUMMARY`: краткая выжимка всей информации в документе.
- `TAGS`: cписок тегов на русском.
- `KEYWORDS`: список ключевых фраз.
- `ВОЗМОЖНЫЕ ВОПРОСЫ`: список вопросов для которых данный документ релевантен.
- `CONTENT`: текстовое наполнение документа.
  
Используй эти данные для:
- Подтверждения фактов.
- Оценки актуальности.
- Формирования ссылки на источник.

### Примеры правильных ответов:

> Вопрос: Когда начинается подача заявок?
> Ответ: Подача заявок начинается 3 апреля и заканчивается 6 мая. Эта информация актуальна на 2025 год. Проверь официальный сайт на предмет обновлений или задай вопрос к кураторам в чате @vse_v_shad.
> Источник: https://shad.yandex.ru/education

> Вопрос: Кто отвечает на вопросы в чате @vse_v_shad?
> Ответ: На вопросы отвечают кураторы: Лёша (@atolstikov), Рита (@Rita_Shadrina), Катя (@pochinkova) и другие.  
> Источник: https://t.me/vse_v_shad/8

> Вопрос: Когда будет экзамен по математике?
> Ответ: вызов функции escalate.


### Контекст:
{context}

### Вопрос:
{question}

### Ответ:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    print("[chain] Creating RAG chain...")
    print(f"[chain] LLM model: {llm.model_name if hasattr(llm, 'model_name') else 'unknown'}")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[chain] RAG chain created successfully")
    return chain
