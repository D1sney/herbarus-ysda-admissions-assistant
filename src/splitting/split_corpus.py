#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split corpus.jsonl into chunks.")
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=["src/ingest/web/data/processed/text/corpus.jsonl"],
        help="Input JSONL file(s) with fields: text, metadata. Can be repeated.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="src/splitting/data/chunks.jsonl",
        help="Output JSONL file with fields: text, metadata.",
    )
    parser.add_argument("--max-chars", type=int, default=1000, help="Max chunk size.")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in chars.")
    return parser.parse_args()


def build_splitter(max_chars: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if max_chars <= 0:
        max_chars = 1000
    if overlap < 0:
        overlap = 0
    if overlap >= max_chars:
        overlap = max_chars // 4
    return RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n"],
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in args.input_jsonl]
    existing_inputs = [p for p in input_paths if p.exists()]
    if not existing_inputs:
        raise SystemExit("No input JSONL files found.")

    documents_data = []
    for path in existing_inputs:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                documents_data.append(row)

    df = pd.DataFrame(documents_data)
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    splitter = build_splitter(args.max_chars, args.overlap)
    total_chunks = 0
    rows = []

    for doc in documents:
        raw_meta = doc.metadata.pop("metadata", None)
        metadata = dict(doc.metadata)
        if isinstance(raw_meta, str):
            try:
                metadata.update(json.loads(raw_meta))
            except json.JSONDecodeError:
                metadata["raw_metadata"] = raw_meta
        elif isinstance(raw_meta, dict):
            metadata.update(raw_meta)

        chunks = splitter.split_text(doc.page_content)
        for idx, chunk in enumerate(chunks):
            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_total"] = len(chunks)
            rows.append(
                {
                    "text": chunk,
                    "metadata": chunk_meta,
                }
            )
            total_chunks += 1

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[split] output={output_path} rows={total_chunks}")


if __name__ == "__main__":
    main()
