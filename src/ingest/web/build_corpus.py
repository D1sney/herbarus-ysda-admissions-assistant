import csv
import json
from pathlib import Path


def build_corpus_csv(
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
        newline="",
    ) as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["text", "metadata"])
        writer.writeheader()
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
            writer.writerow(
                {
                    "text": text,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                }
            )
            kept += 1
            if progress_cb:
                progress_cb(f"[corpus] {kept}")
    if progress_cb:
        progress_cb(f"[corpus] done, rows={kept}")
