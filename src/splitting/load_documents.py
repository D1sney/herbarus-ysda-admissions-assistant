"""Load Document objects from CSV files."""
import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document


def load_documents_from_csv(csv_paths: list[str | Path]) -> list[Document]:
    """Load documents from one or more CSV files.

    Args:
        csv_paths: List of paths to CSV files with columns: text, metadata.

    Returns:
        List of Document objects.
    """
    paths = [Path(p) for p in csv_paths]
    existing = [p for p in paths if p.exists()]
    if not existing:
        raise ValueError(f"No CSV files found: {csv_paths}")

    dataframes = [pd.read_csv(path) for path in existing]
    df = pd.concat(dataframes, ignore_index=True)

    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    # Parse metadata JSON strings if present
    for doc in documents:
        raw_meta = doc.metadata.pop("metadata", None)
        if isinstance(raw_meta, str):
            try:
                parsed = json.loads(raw_meta)
                doc.metadata.update(parsed)
            except json.JSONDecodeError:
                doc.metadata["raw_metadata"] = raw_meta
        elif isinstance(raw_meta, dict):
            doc.metadata.update(raw_meta)

    return documents
