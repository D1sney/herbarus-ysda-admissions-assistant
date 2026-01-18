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


def extract_text_from_json(data, result: list) -> None:
    if isinstance(data, dict):
        content_fields = ['content', 'text', 'title', 'description', 'name', 'alt']
        for key, value in data.items():
            if key in content_fields and isinstance(value, str) and len(value.strip()) > 0:
                stripped = value.strip()
                if (len(stripped) > 3
                    and not stripped.startswith(("http://", "https://", "data:", "#"))
                    and not stripped.startswith("rgba(")
                    and not stripped.startswith("var(")):
                    result.append(stripped)
            extract_text_from_json(value, result)
    elif isinstance(data, list):
        for item in data:
            extract_text_from_json(item, result)
    elif isinstance(data, str) and len(data.strip()) > 0:
        stripped = data.strip()
        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in stripped)
        if (has_cyrillic and len(stripped) > 3
            and not stripped.startswith(("http://", "https://", "data:", "#"))
            and not stripped.startswith("rgba(")
            and not stripped.startswith("var(")):
            result.append(stripped)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    json_texts = []
    for script in soup.find_all("script"):
        script_text = script.string
        if not script_text:
            continue

        script_str = str(script_text)

        try:
            start_positions = []

            markers = ['{"default":', '{"sections":', 'window.__INITIAL_STATE__', 'window.__DATA__']
            for marker in markers:
                idx = script_str.find(marker)
                if idx != -1:
                    brace_pos = script_str.find('{', idx)
                    if brace_pos != -1:
                        start_positions.append(brace_pos)

            if not start_positions:
                for i in range(len(script_str) - 20):
                    if script_str[i:i+11] == '{"default":' or script_str[i:i+12] == '{"sections":':
                        start_positions.append(i)

            for json_start in start_positions[:3]:
                try:
                    brace_count = 0
                    json_end = -1
                    for i in range(json_start, min(json_start + 1000000, len(script_str))):
                        if script_str[i] == '{':
                            brace_count += 1
                        elif script_str[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end > json_start:
                        json_str = script_str[json_start:json_end]
                        data = json.loads(json_str)
                        extract_text_from_json(data, json_texts)
                        break
                except (json.JSONDecodeError, ValueError, IndexError):
                    continue
        except Exception:
            pass

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

    html_text = base.get_text(separator="\n", strip=True)

    all_text = "\n".join(json_texts) + "\n" + html_text
    return clean_text(all_text)


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
