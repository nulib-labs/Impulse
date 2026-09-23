"""Extract tables from a document's OCR output into CSV files.

Reads ``ocr_data.blocks`` from MongoDB ``praxis.colt`` for a given
``impulse_identifier``, finds blocks whose layout label is a table
(``Table`` / ``TableOfContents``), parses their HTML ``<table>`` content, and
writes one CSV per table. The originating page number is encoded in each
CSV filename.

Filename format:
    <identifier>_page<PPPP>_table<N>.csv
e.g. 1928_bankfinancevol1_page0004_table1.csv

Run with:  uv run python scripts/extract_tables.py
Requires:  env IMPULSE_MONGODB_URI.
"""

import os

import certifi
import pandas as pd
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Configuration (overridable via env vars / CLI arg 1 = identifier)
# ---------------------------------------------------------------------------
import sys

IDENTIFIER = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.getenv("IMPULSE_IDENTIFIER", "1928_bankfinancevol1")
)
MAX_PAGES = int(os.getenv("IMPULSE_MAX_PAGES", "50"))
OUT_DIR = os.path.join(
    os.getenv("IMPULSE_OUT_DIR", "moody_sample_output"), "tables"
)

# Layout labels that represent tabular content.
TABLE_LABELS = {"Table", "TableOfContents"}

MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")


def get_colt():
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    return client["praxis"]["colt"]


def html_to_dataframes(html: str) -> list[pd.DataFrame]:
    """Parse an HTML fragment into a list of DataFrames (one per <table>)."""
    if not html or "<table" not in html.lower():
        return []
    try:
        return pd.read_html(html)  # uses lxml
    except ValueError:
        return []  # no tables found by the parser
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] failed to parse table html: {e}")
        return []


def main() -> None:
    if not MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    os.makedirs(OUT_DIR, exist_ok=True)
    colt = get_colt()

    cursor = colt.find(
        {"impulse_identifier": IDENTIFIER, "page_number": {"$lte": MAX_PAGES}}
    ).sort("page_number", 1)

    docs_by_page: dict[int, dict] = {}
    for doc in cursor:
        pn = doc.get("page_number")
        if isinstance(pn, int) and 1 <= pn <= MAX_PAGES:
            docs_by_page[pn] = doc

    total_tables = 0
    pages_with_tables: list[int] = []

    for page in range(1, MAX_PAGES + 1):
        doc = docs_by_page.get(page)
        if doc is None:
            continue

        blocks = (doc.get("ocr_data") or {}).get("blocks", []) or []
        table_idx = 0
        for block in blocks:
            if block.get("label") not in TABLE_LABELS:
                continue
            if block.get("skipped"):
                continue
            dfs = html_to_dataframes(block.get("html", ""))
            for df in dfs:
                if df.empty:
                    continue
                table_idx += 1
                fname = f"{IDENTIFIER}_page{page:04d}_table{table_idx}.csv"
                out_path = os.path.join(OUT_DIR, fname)
                df.to_csv(out_path, index=False)
                total_tables += 1
                print(
                    f"  [page {page}] wrote {out_path} "
                    f"({df.shape[0]} rows x {df.shape[1]} cols)"
                )

        if table_idx:
            pages_with_tables.append(page)

    print(
        f"\nDone. tables={total_tables} "
        f"pages_with_tables={pages_with_tables}"
    )


if __name__ == "__main__":
    main()
