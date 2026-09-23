"""Extract embedded images from the PDFs under ``industrial/``.

Each PDF is a scanned document with one embedded raster image per page.
This extracts every embedded image to ``industrial_images/`` mirroring the
``single_column`` / ``multiple_column`` subdir layout, naming each file with
the source PDF stem and page number.

Filename format:
    <pdf-stem>_page<PPP>.<ext>
e.g. industrial_images/single_column/1920_ex1_page001.png

Run with:  uv run python scripts/extract_pdf_images.py
"""

import glob
import os

import fitz  # PyMuPDF

SRC_ROOT = "industrial"
OUT_ROOT = "industrial_images"


def main() -> None:
    pdfs = sorted(glob.glob(os.path.join(SRC_ROOT, "**", "*.pdf"), recursive=True))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {SRC_ROOT}/")

    total_images = 0
    for pdf_path in pdfs:
        rel_dir = os.path.dirname(os.path.relpath(pdf_path, SRC_ROOT))
        out_dir = os.path.join(OUT_ROOT, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]

        doc = fitz.open(pdf_path)
        pdf_count = 0
        for page_index in range(doc.page_count):
            page = doc[page_index]
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images, start=1):
                xref = img[0]
                base = doc.extract_image(xref)
                ext = base["ext"]
                # Suffix only when a page has more than one embedded image.
                suffix = f"_img{img_idx}" if len(images) > 1 else ""
                fname = f"{stem}_page{page_index + 1:03d}{suffix}.{ext}"
                out_path = os.path.join(out_dir, fname)
                with open(out_path, "wb") as f:
                    f.write(base["image"])
                pdf_count += 1
                total_images += 1
        doc.close()
        print(
            f"  {os.path.relpath(pdf_path):45s} -> {pdf_count} images "
            f"in {out_dir}/"
        )

    print(f"\nDone. extracted {total_images} images from {len(pdfs)} PDFs "
          f"into {OUT_ROOT}/")


if __name__ == "__main__":
    main()
