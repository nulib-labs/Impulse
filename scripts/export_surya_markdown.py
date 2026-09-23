"""Export Surya-2 OCR extraction data from ``praxis.colt`` to Markdown.

For every ``impulse_identifier`` with ``extraction_model == "surya-2"``
(optionally filtered to a prefix, default ``"p"``), dedupes to one doc per
``page_number`` (keeping the most-recently-inserted ``_id`` when duplicates
exist), converts each page's ``ocr_data.blocks[]`` HTML into Markdown in
``reading_order``, and writes one ``.md`` file per identifier.

Run with:  uv run python scripts/export_surya_markdown.py [--limit N] [--overwrite] [--prefix p]
Requires:  env IMPULSE_MONGODB_URI.
"""

import argparse
import os
import re
import sys

import certifi
from bs4 import BeautifulSoup, NavigableString, Tag
from pymongo import MongoClient

MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")
OUT_DIR = os.getenv("IMPULSE_MARKDOWN_OUT_DIR", "extracted_markdown")
EXTRACTION_MODEL = "surya-2"


def get_colt():
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    return client["praxis"]["colt"]


# ---------------------------------------------------------------------------
# HTML -> Markdown conversion
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    return text


def _inline_md(node) -> str:
    """Render inline content (text + b/i/u/sup/sub/br/span/del/math) as Markdown."""
    if isinstance(node, NavigableString):
        return _clean_text(str(node))
    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name == "br":
        return "  \n"
    if name in ("b", "strong"):
        inner = "".join(_inline_md(c) for c in node.children).strip()
        return f"**{inner}**" if inner else ""
    if name in ("i", "em"):
        inner = "".join(_inline_md(c) for c in node.children).strip()
        return f"*{inner}*" if inner else ""
    if name == "u":
        # Markdown has no native underline; use a light HTML fallback.
        inner = "".join(_inline_md(c) for c in node.children).strip()
        return f"<u>{inner}</u>" if inner else ""
    if name == "del":
        inner = "".join(_inline_md(c) for c in node.children).strip()
        return f"~~{inner}~~" if inner else ""
    if name == "sup":
        inner = "".join(_inline_md(c) for c in node.children).strip()
        return f"^{inner}^" if inner else ""
    if name == "sub":
        inner = "".join(_inline_md(c) for c in node.children).strip()
        return f"~{inner}~" if inner else ""
    if name in ("math", "mi", "mn", "mo", "mrow", "msup", "msub", "mfrac"):
        # No real MathML->Markdown conversion; fall back to plain text.
        return node.get_text(" ", strip=True)
    if name == "img":
        alt = node.get("alt", "image")
        return f"![{alt}]"
    if name == "span":
        return "".join(_inline_md(c) for c in node.children)
    # Unknown inline-ish tag: recurse into children.
    return "".join(_inline_md(c) for c in node.children)


def _render_table(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row = [
            "".join(_inline_md(c) for c in cell.children).strip().replace("\n", " ")
            or " "
            for cell in cells
        ]
        if row:
            rows.append(row)

    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for r in rows[1:]:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


def _render_list(list_tag: Tag, ordered: bool) -> str:
    lines = []
    for i, li in enumerate(list_tag.find_all("li", recursive=False), start=1):
        text = "".join(_inline_md(c) for c in li.children).strip()
        prefix = f"{i}." if ordered else "-"
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def html_block_to_markdown(html: str) -> str:
    """Convert a single OCR block's HTML fragment into a Markdown string."""
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")
    parts: list[str] = []

    # Iterate top-level nodes; most blocks are wrapped in a single tag
    # (p, h1-h4, table, ul/ol) but handle bare text / multiple siblings too.
    top_nodes = list(soup.contents)
    if not top_nodes:
        return _clean_text(soup.get_text(" ", strip=True))

    for node in top_nodes:
        if isinstance(node, NavigableString):
            text = _clean_text(str(node)).strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(node, Tag):
            continue

        name = node.name.lower()
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            inner = "".join(_inline_md(c) for c in node.children).strip()
            if inner:
                parts.append(f"{'#' * level} {inner}")
        elif name == "table":
            rendered = _render_table(node)
            if rendered:
                parts.append(rendered)
        elif name == "ul":
            rendered = _render_list(node, ordered=False)
            if rendered:
                parts.append(rendered)
        elif name == "ol":
            rendered = _render_list(node, ordered=True)
            if rendered:
                parts.append(rendered)
        elif name in ("p", "div", "span", "figcaption"):
            inner = "".join(_inline_md(c) for c in node.children).strip()
            if inner:
                parts.append(inner)
        elif name == "hr":
            parts.append("---")
        elif name == "img":
            alt = node.get("alt", "image")
            parts.append(f"![{alt}]")
        else:
            # Fallback: treat unknown top-level tags as inline content.
            inner = "".join(_inline_md(c) for c in node.children).strip()
            if inner:
                parts.append(inner)

    text = "\n\n".join(p for p in parts if p)
    return text.strip()


LABEL_PLACEHOLDER = {
    "Picture": "image",
    "Figure": "figure",
    "Diagram": "diagram",
    "Form": "form",
}


def render_block(block: dict) -> str:
    """Convert one ocr_data block into a Markdown fragment (or placeholder)."""
    label = block.get("label") or ""
    if block.get("skipped") or block.get("error"):
        kind = LABEL_PLACEHOLDER.get(label, label.lower() or "content")
        return f"<!-- {kind} omitted (skipped/error) -->"

    html = block.get("html") or ""
    md = html_block_to_markdown(html)
    if not md:
        return ""
    return md


def render_page(page_number: int, ocr_data: dict) -> str:
    blocks = sorted(
        ocr_data.get("blocks", []) or [],
        key=lambda b: b.get("reading_order") if b.get("reading_order") is not None else 0,
    )
    rendered = [render_block(b) for b in blocks]
    rendered = [r for r in rendered if r]
    body = "\n\n".join(rendered)
    return f"## Page {page_number}\n\n{body}".rstrip()


# ---------------------------------------------------------------------------
# Mongo fetch (dedup by page_number, keep latest _id)
# ---------------------------------------------------------------------------


def discover_identifiers(colt, prefix: str | None) -> list[str]:
    match: dict = {"extraction_model": EXTRACTION_MODEL}
    ids = colt.distinct("impulse_identifier", match)
    ids = [i for i in ids if isinstance(i, str) and i.strip()]
    if prefix:
        ids = [i for i in ids if i.startswith(prefix)]
    return sorted(ids)


def fetch_deduped_pages(colt, identifier: str) -> list[dict]:
    """Return one ocr_data doc per page_number for ``identifier`` (latest _id wins)."""
    pipeline = [
        {
            "$match": {
                "impulse_identifier": identifier,
                "extraction_model": EXTRACTION_MODEL,
            }
        },
        {"$sort": {"page_number": 1, "_id": -1}},
        {"$group": {"_id": "$page_number", "doc": {"$first": "$$ROOT"}}},
        {"$sort": {"_id": 1}},
    ]
    results = list(colt.aggregate(pipeline))
    return [r["doc"] for r in results]


# ---------------------------------------------------------------------------
# Per-identifier export
# ---------------------------------------------------------------------------


def export_identifier(colt, identifier: str, overwrite: bool) -> tuple[int, int]:
    """Write {identifier}.md. Returns (n_pages, n_docs_deduped_away)."""
    out_path = os.path.join(OUT_DIR, f"{identifier}.md")
    if os.path.exists(out_path) and not overwrite:
        return (-1, -1)  # signal "skipped, already exists"

    docs = fetch_deduped_pages(colt, identifier)
    if not docs:
        return (0, 0)

    sections = [f"# {identifier}\n"]
    for doc in docs:
        page_number = doc.get("page_number")
        ocr_data = doc.get("ocr_data") or {}
        sections.append(render_page(page_number, ocr_data))

    text = "\n\n".join(sections)
    text = _BLANKLINES_RE.sub("\n\n\n", text).strip() + "\n"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return (len(docs), 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="p",
        help="Only export identifiers starting with this prefix (default: 'p'). Use '' for all.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N identifiers."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-export even if the .md file already exists."
    )
    parser.add_argument(
        "identifiers",
        nargs="*",
        help="Explicit impulse_identifier(s) to export (overrides discovery).",
    )
    args = parser.parse_args()

    if not MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    colt = get_colt()

    if args.identifiers:
        identifiers = args.identifiers
    else:
        prefix = args.prefix or None
        identifiers = discover_identifiers(colt, prefix)
        if args.limit:
            identifiers = identifiers[: args.limit]

    print(f"Exporting {len(identifiers)} identifiers to {OUT_DIR}/\n")

    n_written = 0
    n_skipped_existing = 0
    n_empty = 0
    total_pages = 0

    for idx, identifier in enumerate(identifiers, start=1):
        n_pages, _ = export_identifier(colt, identifier, args.overwrite)
        if n_pages == -1:
            n_skipped_existing += 1
        elif n_pages == 0:
            n_empty += 1
            print(f"[{idx}/{len(identifiers)}] {identifier}: no docs found")
        else:
            n_written += 1
            total_pages += n_pages

        if idx % 25 == 0 or idx == len(identifiers):
            print(
                f"[{idx}/{len(identifiers)}] progress: written={n_written} "
                f"skipped_existing={n_skipped_existing} empty={n_empty} "
                f"total_pages_written={total_pages}"
            )

    print(
        f"\nDone. written={n_written} skipped_existing={n_skipped_existing} "
        f"empty={n_empty} total_pages_written={total_pages}"
    )


if __name__ == "__main__":
    main()
