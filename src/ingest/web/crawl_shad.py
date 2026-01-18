import hashlib
import json
import time
from collections import deque
from pathlib import Path
import shutil
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


BLACKLIST_PATH_PREFIXES = (
    "/graduates",
    "/intensives",
    "/openday",
    "/special_graduates",
    "/llmscalingweek",
    "/abweek",
    "/bigdwhweek",
    "/cvweek",
    "/sreweek",
    "/gptweek",
)


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return ""
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    parts = parts._replace(scheme="https", netloc=netloc, fragment="", path=path)
    return urlunsplit(parts)


def is_same_domain(url: str, domain: str) -> bool:
    try:
        return urlsplit(url).netloc == domain
    except Exception:
        return False


def is_blacklisted(url: str) -> bool:
    parts = urlsplit(url)
    path = parts.path or "/"
    for prefix in BLACKLIST_PATH_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def url_to_filename(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"page_{digest}.html"


def extract_links(html: str, base_url: str, domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        joined = urljoin(base_url, href)
        normalized = normalize_url(joined)
        if normalized and is_same_domain(normalized, domain) and not is_blacklisted(normalized):
            links.append(normalized)
    return links


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return " ".join(h1.get_text(separator=" ", strip=True).split())
    return ""


def clear_directory(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in out_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def crawl(
    start_urls: list[str],
    out_dir: Path,
    max_depth: int,
    delay: float,
    user_agent: str,
    clear: bool = True,
    progress_cb=None,
) -> list[dict]:
    if clear:
        clear_directory(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    domain = urlsplit(normalize_url(start_urls[0])).netloc
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    queue: deque[tuple[str, int]] = deque()
    seen: set[str] = set()

    for url in start_urls:
        normalized = normalize_url(url)
        if normalized:
            queue.append((normalized, 0))

    index = []

    fetched = 0
    while queue:
        url, depth = queue.popleft()
        if url in seen or depth > max_depth:
            continue
        if is_blacklisted(url):
            continue
        seen.add(url)

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException:
            if progress_cb:
                progress_cb(f"[crawl] skip {url}")
            continue

        html = resp.text
        filename = url_to_filename(url)
        file_path = out_dir / filename
        file_path.write_text(html, encoding="utf-8")

        index.append(
            {
                "url": url,
                "file": str(file_path),
                "title": extract_title(html),
                "status_code": resp.status_code,
            }
        )
        fetched += 1
        if progress_cb:
            progress_cb(f"[crawl] {fetched}: {url}")

        if depth < max_depth:
            for link in extract_links(html, url, domain):
                if link not in seen:
                    queue.append((link, depth + 1))

        if delay > 0:
            time.sleep(delay)

    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if progress_cb:
        progress_cb(f"[crawl] done, pages={len(index)}")
    return index
