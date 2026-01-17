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

    # Check if index exists
    if not force_rebuild and index_path.exists() and (index_path / "index.faiss").exists():
        print(f"[vectorstore] Loading existing index from {index_path}")
        vectorstore = load_vectorstore(index_path, embeddings)
        return vectorstore, True

    # Build new index
    print(f"[vectorstore] Building new index from chunks...")
    documents = load_documents_from_jsonl(chunks_paths)
    print(f"[vectorstore] Loaded {len(documents)} documents")

    vectorstore = create_vectorstore(documents, embeddings)
    print(f"[vectorstore] Index created")

    save_vectorstore(vectorstore, index_path)
    print(f"[vectorstore] Index saved to {index_path}")

    return vectorstore, False
