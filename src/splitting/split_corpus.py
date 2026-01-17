#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split corpus.csv into chunks.")
    parser.add_argument(
        "--input-csv",
        action="append",
        default=["src/ingest/web/data/processed/text/corpus.csv"],
        help="Input CSV(s) with columns: text, metadata. Can be repeated.",
    )
    parser.add_argument(
        "--output-csv",
        default="src/splitting/data/chunks.csv",
        help="Output CSV with columns: text, metadata.",
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
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in args.input_csv]
    existing_inputs = [p for p in input_paths if p.exists()]
    if not existing_inputs:
        raise SystemExit("No input CSV files found.")

    dataframes = [pd.read_csv(path) for path in existing_inputs]
    df = pd.concat(dataframes, ignore_index=True)
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
                    "metadata": json.dumps(chunk_meta, ensure_ascii=False),
                }
            )
            total_chunks += 1

    out_df = pd.DataFrame(rows, columns=["text", "metadata"])
    out_df.to_csv(output_path, index=False)

    print(f"[split] output={output_path} rows={total_chunks}")


if __name__ == "__main__":
    main()
