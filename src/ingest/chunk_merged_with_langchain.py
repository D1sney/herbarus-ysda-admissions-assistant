import json
import re
from pathlib import Path

from langchain_core.documents import Document

INPUT_JSONL = Path("/Users/shchsergey/programming/herbarus-ysda-rag/src/ingest/merged/knowledge_base.jsonl")
OUTPUT_JSONL = Path("/Users/shchsergey/programming/herbarus-ysda-rag/src/ingest/merged/knowledge_base_chunks.jsonl")

SEPARATOR_CANON = "--- НАЧАЛО ЧАНКА ---"
SPLIT_RE = re.compile(r"(?:\r?\n)?\s*---\s*НАЧАЛО\s*ЧАНКА\s*---\s*(?:\r?\n)?", re.UNICODE)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    docs = []
    for obj in iter_jsonl(INPUT_JSONL):
        docs.append(Document(page_content=obj["text"], metadata=obj.get("metadata", {})))

    print("Loaded docs:", len(docs))

    out_docs = []
    for doc_i, d in enumerate(docs):
        text = (d.page_content or "").strip()
        if not text:
            continue

        parts = [p.strip() for p in SPLIT_RE.split(text) if p and p.strip()]

        if not parts:
            parts = [text]

        for chunk_i, p in enumerate(parts):
            chunk_text = f"{SEPARATOR_CANON}\n\n{p}"

            md = dict(d.metadata)
            md["doc_index"] = doc_i
            md["chunk_index"] = chunk_i

            out_docs.append(Document(page_content=chunk_text, metadata=md))

    print("Chunks produced:", len(out_docs))

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for d in out_docs:
            out.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

    print("Saved:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
