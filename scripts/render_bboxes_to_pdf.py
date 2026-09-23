"""Render Surya layout bboxes (+ OCR overlay) for one or more documents and
assemble each document's annotated pages into a single multi-page PDF.

Reuses the page-drawing logic from ``scripts/render_bboxes.py`` (S3 image
download, layout-bbox drawing, OCR text overlay) but writes one PDF per
``impulse_identifier`` instead of separate PNGs.

By default, targets every ``impulse_identifier`` in ``praxis.colt`` matching
the pattern ``19??_ex*`` (e.g. ``1920_ex1``, ``1928_ex5``).

Run with:  uv run python scripts/render_bboxes_to_pdf.py
Requires:  env IMPULSE_MONGODB_URI, AWS profile "impulse".
"""

import os
import random
import re
import sys

import render_bboxes as rb

OUT_DIR = os.getenv("IMPULSE_OUT_DIR", "moody_sample_output")
IDENTIFIER_PATTERN = re.compile(r"^19\d\d_ex")


def discover_identifiers() -> list[str]:
    """Find all impulse_identifiers in praxis.colt matching ``19**_ex*``."""
    colt = rb.get_colt()
    ids = colt.distinct("impulse_identifier")
    matches = sorted(
        i for i in ids if isinstance(i, str) and IDENTIFIER_PATTERN.match(i)
    )
    return matches


def render_random_sample_to_pdf(n: int, seed: int | None = None) -> None:
    """Pick ``n`` random (identifier, page) pairs across matching docs and
    render them all into a single sample PDF for a quick sanity check."""
    colt = rb.get_colt()
    font = rb.load_font()

    identifiers = discover_identifiers()
    all_pairs: list[tuple[str, dict]] = []
    for identifier in identifiers:
        for doc in colt.find({"impulse_identifier": identifier}):
            pn = doc.get("page_number")
            if isinstance(pn, int):
                all_pairs.append((identifier, doc))

    print(f"Found {len(all_pairs)} total page docs across {len(identifiers)} identifiers")

    rng = random.Random(seed)
    sample = rng.sample(all_pairs, min(n, len(all_pairs)))
    # Sort for a stable, readable ordering in the output PDF.
    sample.sort(key=lambda pair: (pair[0], pair[1].get("page_number")))

    rendered_images = []
    for identifier, doc in sample:
        page = doc.get("page_number")
        rb.IDENTIFIER = identifier
        layout_data = doc.get("layout_data") or {}
        ocr_data = doc.get("ocr_data") or {}
        n_boxes = len(layout_data.get("bboxes", []) or [])

        img = rb.download_raw_image(page)
        if img is None:
            print(f"  [{identifier} page {page}] no image, skipping")
            continue

        img = rb.draw_page(img, layout_data, ocr_data, font)
        rendered_images.append(img.convert("RGB"))
        print(f"  [{identifier} page {page}] rendered ({n_boxes} boxes)")

    if not rendered_images:
        print("No pages rendered, aborting.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"sample_{n}_random_pages.pdf")
    first, rest = rendered_images[0], rendered_images[1:]
    first.save(out_path, save_all=True, append_images=rest)
    print(f"\nWrote {out_path} ({len(rendered_images)} pages)")


def render_identifier_to_pdf(identifier: str) -> None:
    """Render all pages for ``identifier`` and save as one PDF."""
    rb.IDENTIFIER = identifier  # module-level global used by download_raw_image

    colt = rb.get_colt()
    font = rb.load_font()

    cursor = colt.find({"impulse_identifier": identifier}).sort("page_number", 1)
    docs_by_page: dict[int, dict] = {}
    for doc in cursor:
        pn = doc.get("page_number")
        if isinstance(pn, int):
            docs_by_page[pn] = doc

    if not docs_by_page:
        print(f"[{identifier}] no page docs found, skipping")
        return

    pages = sorted(docs_by_page.keys())
    print(f"[{identifier}] {len(pages)} page docs found: {pages}")

    rendered_images = []
    missing_image_pages: list[int] = []
    no_box_pages: list[int] = []

    for page in pages:
        doc = docs_by_page[page]
        layout_data = doc.get("layout_data") or {}
        ocr_data = doc.get("ocr_data") or {}
        if not layout_data.get("bboxes"):
            no_box_pages.append(page)

        img = rb.download_raw_image(page)
        if img is None:
            missing_image_pages.append(page)
            continue

        img = rb.draw_page(img, layout_data, ocr_data, font)
        rendered_images.append(img.convert("RGB"))
        n_boxes = len(layout_data.get("bboxes", []) or [])
        print(f"  [{identifier} page {page}] rendered ({n_boxes} boxes)")

    if not rendered_images:
        print(f"[{identifier}] no pages rendered, skipping PDF write")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{identifier}.pdf")
    first, rest = rendered_images[0], rendered_images[1:]
    first.save(out_path, save_all=True, append_images=rest)
    print(
        f"[{identifier}] wrote {out_path} "
        f"({len(rendered_images)} pages; missing_images={missing_image_pages} "
        f"no_boxes={no_box_pages})\n"
    )


def main() -> None:
    if not rb.MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    args = sys.argv[1:]
    if args and args[0] == "--sample":
        n = int(args[1]) if len(args) > 1 else 10
        render_random_sample_to_pdf(n)
        return

    identifiers = args or discover_identifiers()
    print(f"Rendering {len(identifiers)} identifiers: {identifiers}\n")

    for identifier in identifiers:
        render_identifier_to_pdf(identifier)


if __name__ == "__main__":
    main()
