from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever


def create_retriever(
    vectorstore: FAISS,
    search_type: str = "similarity",
    k: int = 20,
    score_threshold: float | None = None,
) -> BaseRetriever:
    """Create a retriever from FAISS vectorstore.

    Args:
        vectorstore: FAISS vectorstore instance.
        search_type: Type of search ("similarity" or "similarity_score_threshold").
        k: Number of documents to return.
        score_threshold: Minimum score threshold (only for similarity_score_threshold).

    Returns:
        BaseRetriever instance.
    """
    kwargs = {
        "search_type": search_type,
        "k": k,
    }

    if search_type == "similarity_score_threshold" and score_threshold is not None:
        kwargs["score_threshold"] = score_threshold

    return vectorstore.as_retriever(**kwargs)
