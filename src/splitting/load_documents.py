"""Load Document objects from JSONL files."""
import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document


def load_documents_from_jsonl(jsonl_paths: list[str | Path]) -> list[Document]:
    """Load documents from one or more JSONL files.

    Args:
        jsonl_paths: List of paths to JSONL files with fields: text, metadata.

    Returns:
        List of Document objects.
    """
    paths = [Path(p) for p in jsonl_paths]
    existing = [p for p in paths if p.exists()]
    if not existing:
        raise ValueError(f"No JSONL files found: {jsonl_paths}")

    documents_data = []
    for path in existing:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                documents_data.append(row)

    df = pd.DataFrame(documents_data)
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    for doc in documents:
        raw_meta = doc.metadata.pop("metadata", None)
        if isinstance(raw_meta, dict):
            doc.metadata.update(raw_meta)
        elif isinstance(raw_meta, str):
            try:
                parsed = json.loads(raw_meta)
                doc.metadata.update(parsed)
            except json.JSONDecodeError:
                doc.metadata["raw_metadata"] = raw_meta

    return documents
