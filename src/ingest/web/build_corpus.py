import json
from pathlib import Path


def build_corpus_jsonl(
    input_path: Path,
    output_path: Path,
    min_chars: int = 200,
    progress_cb=None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "w",
        encoding="utf-8",
    ) as out_f:
        for line in in_f:
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < min_chars:
                continue
            metadata = {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "source_file": row.get("source_file", ""),
            }
            out_row = {
                "text": text,
                "metadata": metadata,
            }
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1
            if progress_cb:
                progress_cb(f"[corpus] {kept}")
    if progress_cb:
        progress_cb(f"[corpus] done, rows={kept}")
