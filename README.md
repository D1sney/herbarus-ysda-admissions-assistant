# herbarus-ysda-rag

Baseline knowledge base ingestion for the YSDA admissions assistant.

## Structure
- `src/ingest/web/` — crawl, text extraction, corpus build
- `src/ingest/web/data/raw/html/` — raw HTML pages
- `src/ingest/web/data/processed/text/` — extracted text and corpus

## Run (baseline)
- `python3 src/ingest/web/build_knowledge_base.py --max-depth 1`
