import json
from pathlib import Path

from bs4 import BeautifulSoup


STRIP_TAGS = {"script", "style", "noscript", "header", "footer", "nav", "svg"}


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = " ".join(line.strip().split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in STRIP_TAGS:
        for node in soup.find_all(tag):
            node.decompose()

    main = soup.find("main")
    if main:
        base = main
    elif soup.body:
        base = soup.body
    else:
        base = soup

    text = base.get_text(separator="\n", strip=True)
    return clean_text(text)


def build_pages_jsonl(
    index: list[dict],
    out_path: Path,
    min_chars: int = 200,
    progress_cb=None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(index, start=1):
            html_path = Path(row["file"])
            if not html_path.exists():
                continue
            html = html_path.read_text(encoding="utf-8")
            text = extract_text(html)
            if len(text) < min_chars:
                continue
            out_row = {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "text": text,
                "source_file": str(html_path),
            }
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1
            if progress_cb:
                progress_cb(f"[text] {i}/{len(index)}")
    if progress_cb:
        progress_cb(f"[text] done, pages={kept}")
