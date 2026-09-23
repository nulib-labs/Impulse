"""Render Surya layout bounding boxes for the first N pages of a document.

Fetches OCR/layout data from MongoDB ``praxis.colt`` for a given
``impulse_identifier``, downloads the associated raw_images from S3, draws the
``layout_data`` element boxes on top, and writes annotated images to
``moody_sample_output/``.

Run with:  uv run python scripts/render_bboxes.py
Requires:  env IMPULSE_MONGODB_URI, AWS profile "impulse".
"""

import os
import re
from html import unescape
from io import BytesIO

import boto3
import certifi
from PIL import Image, ImageDraw, ImageFont
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
BUCKET = "nu-impulse-data"
OUT_DIR = os.getenv("IMPULSE_OUT_DIR", "moody_sample_output")

MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

# Draw the recognized OCR text (ocr_data.blocks[].html) over each block region.
DRAW_OCR = True
OCR_BOX_COLOR = (0, 0, 0)
OCR_TEXT_COLOR = (200, 30, 30)

# Label -> RGB color for layout element outlines.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "Text": (31, 119, 180),
    "Title": (214, 39, 40),
    "SectionHeader": (255, 127, 14),
    "Section-header": (255, 127, 14),
    "PageHeader": (148, 103, 189),
    "Page-header": (148, 103, 189),
    "PageFooter": (140, 86, 75),
    "Page-footer": (140, 86, 75),
    "Table": (44, 160, 44),
    "Picture": (227, 119, 194),
    "Figure": (188, 189, 34),
    "Caption": (23, 190, 207),
    "ListItem": (255, 187, 120),
    "List-item": (255, 187, 120),
    "Formula": (152, 223, 138),
    "Footnote": (197, 176, 213),
    "TextInlineMath": (174, 199, 232),
}
DEFAULT_COLOR = (127, 127, 127)


def get_colt():
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    return client["praxis"]["colt"]


def download_raw_image(page_number: int) -> Image.Image | None:
    """Download the raw_images JPG the model ran on for a given page."""
    project, accession = IDENTIFIER.split("_", 1)
    filename = f"{project}_{accession}_{page_number:010d}.jpg"
    key = f"{project}/{accession}/raw_images/{filename}"
    session = boto3.Session(profile_name="impulse")
    s3 = session.client("s3")
    try:
        buf = BytesIO()
        s3.download_fileobj(BUCKET, key, buf)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"  [page {page_number}] no S3 image at {key}: {e}")
        return None


def load_font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _load_font_sized(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


# Descending font sizes used to fit OCR text inside each block.
OCR_FONTS = [(_load_font_sized(s), s) for s in (48, 40, 34, 28, 24, 20, 16, 13, 10, 8)]


def scale_factors(image_bbox, img_w, img_h) -> tuple[float, float]:
    """image_bbox is [x0, y0, x1, y1] in the model's coordinate space."""
    if not image_bbox or len(image_bbox) != 4:
        return 1.0, 1.0
    bw = image_bbox[2] - image_bbox[0]
    bh = image_bbox[3] - image_bbox[1]
    sx = img_w / bw if bw else 1.0
    sy = img_h / bh if bh else 1.0
    return sx, sy


def html_to_text(html: str) -> str:
    """Strip tags/entities from a block's html into plain text."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def poly_to_points(box: dict, sx: float, sy: float):
    """Return scaled polygon points for a box (falls back to bbox)."""
    polygon = box.get("polygon")
    if polygon and len(polygon) >= 3:
        return [(p[0] * sx, p[1] * sy) for p in polygon]
    bb = box.get("bbox")
    if not bb or len(bb) != 4:
        return None
    x0, y0, x1, y1 = bb
    return [
        (x0 * sx, y0 * sy),
        (x1 * sx, y0 * sy),
        (x1 * sx, y1 * sy),
        (x0 * sx, y1 * sy),
    ]


def _text_size(draw, text: str, font):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:  # noqa: BLE001
        return 8 * len(text), 14


def wrap_text(draw, text: str, font, max_w: float) -> list[str]:
    """Greedy word-wrap ``text`` so each line fits within ``max_w``."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        w, _ = _text_size(draw, trial, font)
        if w <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def fit_wrapped(draw, text: str, max_w: float, max_h: float):
    """Choose the largest font whose word-wrapped text fits the block.

    Returns ``(font, lines, line_height)``.
    """
    best = None
    for font, _size in OCR_FONTS:
        lines = wrap_text(draw, text, font, max_w)
        if not lines:
            continue
        _, sample_h = _text_size(draw, "Ag", font)
        line_h = int(sample_h * 1.15) + 1
        total_h = line_h * len(lines)
        # Widest line must fit horizontally too.
        widest = max(_text_size(draw, ln, font)[0] for ln in lines)
        if total_h <= max_h and widest <= max_w:
            return font, lines, line_h
        best = (font, lines, line_h)  # smallest so far as fallback
    return best if best else (OCR_FONTS[-1][0], [text], 10)


def draw_ocr(img: Image.Image, ocr_data: dict, base_draw) -> None:
    """Overlay recognized OCR text on a semi-transparent strip per block."""
    img_w, img_h = img.size
    sx, sy = scale_factors(ocr_data.get("image_bbox"), img_w, img_h)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    for block in ocr_data.get("blocks", []) or []:
        if block.get("skipped"):
            continue
        text = html_to_text(block.get("html", ""))
        if not text:
            continue
        pts = poly_to_points(block, sx, sy)
        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)

        # Thin outline around the OCR block on the main image.
        base_draw.polygon(pts, outline=OCR_BOX_COLOR, width=1)

        pad = 2.0
        font, lines, line_h = fit_wrapped(
            odraw, text, bw - 2 * pad, bh - 2 * pad
        )
        # Semi-transparent white backing so text is legible over the scan.
        odraw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 205))
        ty = y0 + pad
        for line in lines:
            odraw.text(
                (x0 + pad, ty),
                line,
                fill=OCR_TEXT_COLOR + (255,),
                font=font,
            )
            ty += line_h

    img.alpha_composite(overlay)


def draw_page(img: Image.Image, layout_data: dict, ocr_data: dict, font) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    sx, sy = scale_factors(layout_data.get("image_bbox"), img_w, img_h)

    if DRAW_OCR and ocr_data:
        draw_ocr(img, ocr_data, draw)

    for box in layout_data.get("bboxes", []) or []:
        label = box.get("label", "?")
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        polygon = box.get("polygon")
        if polygon and len(polygon) >= 3:
            pts = [(p[0] * sx, p[1] * sy) for p in polygon]
        else:
            bb = box.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x0, y0, x1, y1 = bb
            pts = [
                (x0 * sx, y0 * sy),
                (x1 * sx, y0 * sy),
                (x1 * sx, y1 * sy),
                (x0 * sx, y1 * sy),
            ]
        draw.polygon(pts, outline=color, width=3)

        # Label strip: "<label> #<position>"
        position = box.get("position")
        tag = f"{label}" + (f" #{position}" if position is not None else "")
        tx, ty = pts[0]
        try:
            l, t, r, b = draw.textbbox((0, 0), tag, font=font)
            tw, th = r - l, b - t
        except Exception:  # noqa: BLE001
            tw, th = 8 * len(tag), 14
        strip_y = max(0, ty - th - 2)
        draw.rectangle([tx, strip_y, tx + tw + 4, strip_y + th + 2], fill=color)
        draw.text((tx + 2, strip_y + 1), tag, fill=(255, 255, 255), font=font)

    return img


def main() -> None:
    if not MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    os.makedirs(OUT_DIR, exist_ok=True)
    colt = get_colt()
    font = load_font()

    cursor = colt.find(
        {"impulse_identifier": IDENTIFIER, "page_number": {"$lte": MAX_PAGES}}
    ).sort("page_number", 1)

    docs_by_page: dict[int, dict] = {}
    for doc in cursor:
        pn = doc.get("page_number")
        if isinstance(pn, int) and 1 <= pn <= MAX_PAGES:
            docs_by_page[pn] = doc

    print(
        f"Found {len(docs_by_page)} page docs for {IDENTIFIER} "
        f"(page_number <= {MAX_PAGES})"
    )

    rendered = 0
    no_data_pages: list[int] = []
    missing_image_pages: list[int] = []

    for page in range(1, MAX_PAGES + 1):
        doc = docs_by_page.get(page)
        if doc is None:
            no_data_pages.append(page)
            continue

        layout_data = doc.get("layout_data") or {}
        ocr_data = doc.get("ocr_data") or {}
        if not layout_data.get("bboxes"):
            print(f"  [page {page}] doc has no layout bboxes")

        img = download_raw_image(page)
        if img is None:
            missing_image_pages.append(page)
            continue

        img = draw_page(img, layout_data, ocr_data, font)
        out_path = os.path.join(
            OUT_DIR, f"{IDENTIFIER}_{page:010d}.png"
        )
        img.convert("RGB").save(out_path)
        rendered += 1
        n_boxes = len(layout_data.get("bboxes", []) or [])
        print(f"  [page {page}] wrote {out_path} ({n_boxes} boxes)")

    print(
        f"\nDone. rendered={rendered} "
        f"no_data_pages={no_data_pages} "
        f"missing_image_pages={missing_image_pages}"
    )


if __name__ == "__main__":
    main()
