#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from build_corpus import build_corpus_csv
from crawl_shad import crawl
from extract_text import build_pages_jsonl


def prepare_dirs(base_dir: Path, clear: bool) -> dict[str, Path]:
    if clear and base_dir.exists():
        shutil.rmtree(base_dir)
    raw_dir = base_dir / "raw" / "html"
    processed_dir = base_dir / "processed" / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "processed": processed_dir}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl shad.yandex.ru and build CSV corpus.")
    parser.add_argument(
        "--start-url",
        action="append",
        default=["https://shad.yandex.ru/"],
        help="Seed URL(s). Can be repeated.",
    )
    parser.add_argument(
        "--data-dir",
        default="src/ingest/web/data",
        help="Base directory for raw/processed data.",
    )
    parser.add_argument("--max-depth", type=int, default=1, help="Max crawl depth.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (sec).")
    parser.add_argument("--min-chars", type=int, default=200, help="Skip pages shorter than this.")
    parser.add_argument(
        "--user-agent",
        default="ysda-admissions-assistant/0.1 (+https://shad.yandex.ru/)",
        help="User-Agent header.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not очистить data-dir before run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.data_dir)
    dirs = prepare_dirs(base_dir, clear=not args.no_clear)

    def progress(message: str) -> None:
        print(message)

    index = crawl(
        start_urls=args.start_url,
        out_dir=dirs["raw"],
        max_depth=args.max_depth,
        delay=args.delay,
        user_agent=args.user_agent,
        clear=False,
        progress_cb=progress,
    )

    pages_path = dirs["processed"] / "pages.jsonl"
    build_pages_jsonl(
        index=index,
        out_path=pages_path,
        min_chars=args.min_chars,
        progress_cb=progress,
    )

    corpus_path = dirs["processed"] / "corpus.csv"
    build_corpus_csv(
        input_path=pages_path,
        output_path=corpus_path,
        min_chars=args.min_chars,
        progress_cb=progress,
    )
    print(f"[output] {pages_path}")
    print(f"[output] {corpus_path}")


if __name__ == "__main__":
    main()
