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
    if prompt_template is None:
        prompt_template = """Ты - ассистент по поступлению в Школу анализа данных (ШАД).

На основе следующей информации из базы знаний дай точный и фактический ответ на вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

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
