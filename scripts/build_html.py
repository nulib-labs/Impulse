"""Combine the OCR HTML of the first N pages into a single HTML file.

Reads ``ocr_data.blocks`` from MongoDB ``praxis.colt`` for a given
``impulse_identifier``, orders each page's blocks by ``reading_order``,
concatenates their ``html``, and wraps all pages into one standalone HTML
document with per-page section headers.

Run with:  uv run python scripts/build_html.py
Requires:  env IMPULSE_MONGODB_URI.
"""

import os

import certifi
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
OUT_DIR = os.getenv("IMPULSE_OUT_DIR", "moody_sample_output")
OUT_PATH = os.path.join(OUT_DIR, f"{IDENTIFIER}_pages1-{MAX_PAGES}.html")

MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

STYLE = """
  body { font-family: Georgia, 'Times New Roman', serif; max-width: 820px;
         margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }
  .page { border-bottom: 2px solid #ccc; padding: 2rem 0; }
  .page-label { font-family: system-ui, sans-serif; font-size: 0.8rem;
                color: #888; letter-spacing: 0.05em; text-transform: uppercase;
                margin-bottom: 1rem; }
  table { border-collapse: collapse; margin: 1rem 0; width: 100%; }
  td, th { border: 1px solid #bbb; padding: 4px 8px; vertical-align: top; }
  h1, h2, h3 { line-height: 1.2; }
  img { max-width: 100%; }
"""


def get_colt():
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    return client["praxis"]["colt"]


def page_html(doc: dict) -> str:
    blocks = (doc.get("ocr_data") or {}).get("blocks", []) or []
    ordered = sorted(
        blocks, key=lambda b: b.get("reading_order", 1_000_000)
    )
    parts: list[str] = []
    for b in ordered:
        if b.get("skipped"):
            continue
        html = b.get("html") or ""
        if html.strip():
            parts.append(html)
    return "\n".join(parts)


def main() -> None:
    if not MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    colt = get_colt()

    cursor = colt.find(
        {"impulse_identifier": IDENTIFIER, "page_number": {"$lte": MAX_PAGES}}
    ).sort("page_number", 1)

    docs_by_page: dict[int, dict] = {}
    for doc in cursor:
        pn = doc.get("page_number")
        if isinstance(pn, int) and 1 <= pn <= MAX_PAGES:
            docs_by_page[pn] = doc

    sections: list[str] = []
    included = 0
    missing_pages: list[int] = []
    for page in range(1, MAX_PAGES + 1):
        doc = docs_by_page.get(page)
        if doc is None:
            missing_pages.append(page)
            continue
        body = page_html(doc)
        sections.append(
            f'<section class="page" id="page-{page}">\n'
            f'  <div class="page-label">Page {page}</div>\n'
            f"{body}\n"
            f"</section>"
        )
        included += 1

    document = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{IDENTIFIER} — pages 1–{MAX_PAGES}</title>\n"
        f"  <style>{STYLE}</style>\n"
        "</head>\n<body>\n"
        f"  <h1>{IDENTIFIER} — pages 1–{MAX_PAGES}</h1>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>\n"
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(document)

    print(
        f"Done. wrote {OUT_PATH} "
        f"(pages_included={included} missing_pages={missing_pages})"
    )


if __name__ == "__main__":
    main()
