"""FAISS vectorstore creation and persistence."""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_vectorstore(
    documents: list[Document],
    embeddings: Embeddings,
) -> FAISS:
    """Create FAISS vectorstore from documents.

    Args:
        documents: List of Document objects to index.
        embeddings: Embeddings model to use.

    Returns:
        FAISS vectorstore instance.
    """
    return FAISS.from_documents(documents, embeddings)


def save_vectorstore(vectorstore: FAISS, save_path: str | Path) -> None:
    """Save FAISS vectorstore to disk.

    Args:
        vectorstore: FAISS vectorstore to save.
        save_path: Directory path where to save the index.
    """
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_vectorstore(
    load_path: str | Path,
    embeddings: Embeddings,
) -> FAISS:
    """Load FAISS vectorstore from disk.

    Args:
        load_path: Directory path where the index is saved.
        embeddings: Embeddings model (must match the one used for creation).

    Returns:
        FAISS vectorstore instance.
    """
    return FAISS.load_local(str(load_path), embeddings, allow_dangerous_deserialization=True)
