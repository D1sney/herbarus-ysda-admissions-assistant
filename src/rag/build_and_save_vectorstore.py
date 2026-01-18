"""Build and save vectorstore from corpus."""
import os
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from rag.vectorstore import create_vectorstore, load_vectorstore, save_vectorstore
from splitting.load_documents import load_documents_from_jsonl


def build_or_load_vectorstore(
    chunks_paths: list[str | Path],
    index_path: str | Path,
    embeddings: OpenAIEmbeddings,
    force_rebuild: bool = False,
) -> tuple:
    """Build vectorstore from chunks or load existing one.

    Args:
        chunks_paths: List of paths to chunks JSONL files (from splitting).
        index_path: Path where to save/load the vectorstore index.
        embeddings: Embeddings model to use.
        force_rebuild: If True, rebuild even if index exists.

    Returns:
        Tuple of (vectorstore, was_loaded) where was_loaded is True if loaded from disk.
    """
    index_path = Path(index_path)

    if not force_rebuild and index_path.exists() and (index_path / "index.faiss").exists():
        vectorstore = load_vectorstore(index_path, embeddings)
        return vectorstore, True

    documents = load_documents_from_jsonl(chunks_paths)
    vectorstore = create_vectorstore(documents, embeddings)
    save_vectorstore(vectorstore, index_path)

    return vectorstore, False
