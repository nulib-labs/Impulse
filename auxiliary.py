from uuid import uuid4
import PIL
from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
import certifi
from fireworks.core.firework import FWAction
from marker.renderers.json import JSONOutput
import woolworm
import cv2
import numpy as np
import tempfile
import io
from PIL import Image
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from tqdm import tqdm
import math
from typing import Union, Tuple
from deskew import determine_skew
import lxml.etree as ET
from loguru import logger
from pymongo import MongoClient
import boto3
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

conn_str: str
conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))

url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")

s3 = boto3.client("s3")


def rotate(
    image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat, (int(round(height)), int(round(width))), borderValue=background
    )


def process_historical_document(image_np):
    """
    Process a historical document using Otsu binarization.
    Returns the binarized image.
    """

    # Convert to grayscale if RGB
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Denoise
    dst = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Otsu’s thresholding (black text on white)
    otsu_thresh, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = otsu_thresh * 1.2
    _, binary = cv2.threshold(dst, adjusted_thresh, 255, cv2.THRESH_BINARY)
    # (Optional) Diagnostic printouts — can be removed
    foreground_ratio = np.sum(binary == 0) / binary.size  # proportion of dark pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    large_components = sizes[sizes > 50]

    print(f"→ Foreground ratio: {foreground_ratio:.3f}")
    print(f"→ Large components: {len(large_components)}")

    # Return just the binarized image
    return binary


def get_mongo_client_kwargs():
    return {
        "tls": True,
        "tlsCAFile": certifi.where(),
    }


lp = LaunchPad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    name="fireworks",
    mongoclient_kwargs=get_mongo_client_kwargs(),
)

fp = FilePad(
    host=conn_str,
    port=27017,
    name="fireworks",
    uri_mode=True,
    mongoclient_kwargs=get_mongo_client_kwargs(),
)


def convert_mets_to_yml(*args):
    """
    FireWorks task function: Convert a METS XML file from GridFS into a YAML file
    following HathiTrust ingest specs, and save the YAML back to GridFS.
    args[0] = gfs_id of the METS XML file in GridFS
    """
    gfs_id = args[0]
    accession_number = args[1]
    # === Step 1: Get XML contents from GridFS ===
    file_contents, doc = fp.get_file_by_id(gfs_id)
    print(file_contents)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_xml:
        tmp_xml.write(file_contents)
        tmp_xml_path = tmp_xml.name

    # === Step 2: Parse the XML ===
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(tmp_xml_path, parser)
        root = tree.getroot()
    except ET.XMLSyntaxError as e:
        print(f"Error: Invalid XML file: {e}")
        raise RuntimeError("Invalid METS XML file") from e

    ns = {
        "xmlns": "http://www.loc.gov/METS/",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    def find_filename_by_file_id(file_id):
        node = root.xpath(f"//xmlns:file[@ID='{file_id}']/xmlns:FLocat", namespaces=ns)
        if not node:
            return None
        href = node[0].get("{http://www.w3.org/1999/xlink}href", "")
        return href[7:] if href.startswith("file://") else href

    # === Step 3: Extract header information ===
    mets_hdr = root.xpath("//xmlns:metsHdr", namespaces=ns)
    capture_date = mets_hdr[0].get("CREATEDATE") + "-06:00" if mets_hdr else None

    suprascan = False
    scanning_order_rtl = False
    reading_order_rtl = False
    resolution = 400

    yaml_lines = []
    yaml_lines.append(f"capture_date: {capture_date}")
    if suprascan:
        yaml_lines.append("scanner_make: SupraScan")
        yaml_lines.append("scanner_model: Quartz A1")
    else:
        yaml_lines.append("scanner_make: Kirtas")
        yaml_lines.append("scanner_model: APT 1200")
    yaml_lines.append(
        'scanner_user: "Northwestern University Library: Repository & Digital Curation"'
    )
    yaml_lines.append(f"contone_resolution_dpi: {resolution}")
    yaml_lines.append(f"image_compression_date: {capture_date}")
    yaml_lines.append("image_compression_agent: northwestern")
    yaml_lines.append('image_compression_tool: ["LIMB v4.5.0.0"]')
    yaml_lines.append(
        f"scanning_order: {'right-to-left' if scanning_order_rtl else 'left-to-right'}"
    )
    yaml_lines.append(
        f"reading_order: {'right-to-left' if reading_order_rtl else 'left-to-right'}"
    )
    yaml_lines.append("pagedata:")

    # === Step 4: Logical structMap page iteration ===
    logical_pages = root.xpath(
        '//xmlns:structMap[@TYPE="logical"]//xmlns:div[@TYPE="page"]', namespaces=ns
    )

    for element in logical_pages:
        fileptr = element.xpath(
            "./xmlns:fptr[starts-with(@FILEID, 'JP2')]", namespaces=ns
        )
        if not fileptr:
            continue
        file_id = fileptr[0].get("FILEID")
        filename = find_filename_by_file_id(file_id)
        if not filename:
            continue

        parent = element.getparent()
        parent_label = parent.get("LABEL", "")
        parent_type = parent.get("TYPE", "")
        orderlabel = element.get("ORDERLABEL", "")
        line = None

        # Label logic
        if element == parent[0]:
            if (
                parent_label == "Cover"
                and parent_type == "cover"
                and parent == logical_pages[0].getparent()
            ):
                label = "FRONT_COVER"
                line = f'{filename}: {{ label: "{label}" }}'
            elif parent_label == "Front Matter" and orderlabel:
                line = f'{filename}: {{ orderlabel: "{orderlabel}" }}'
            elif parent_label == "Title":
                label = "TITLE"
                line = (
                    f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                    if orderlabel
                    else f'{filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Contents":
                label = "TABLE_OF_CONTENTS"
                line = (
                    f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                    if orderlabel
                    else f'{filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Preface":
                label = "PREFACE"
                line = (
                    f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                    if orderlabel
                    else f'{filename}: {{ label: "{label}" }}'
                )
            elif parent_label.startswith("Chapter") or parent_label == "Appendix":
                label = "CHAPTER_START"
                line = (
                    f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                    if orderlabel
                    else f'{filename}: {{ label: "{label}" }}'
                )
            elif parent_label in ("Notes", "Bibliography"):
                label = "REFERENCES"
                line = (
                    f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                    if orderlabel
                    else f'{filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Index":
                label = "INDEX"
                line = (
                    f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                    if orderlabel
                    else f'{filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Cover" and parent_type == "cover":
                label = "BACK_COVER"
                line = f'{filename}: {{ label: "{label}" }}'
        else:
            if orderlabel:
                line = f'{filename}: {{ orderlabel: "{orderlabel}" }}'

        if line:
            yaml_lines.append("    " + line)

    # === Step 5: Write YAML to temp file ===

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yml") as tmp_yml:
        tmp_yml.write("\n".join(yaml_lines).encode("utf-8"))
        tmp_yml_path = tmp_yml.name

    # === Step 6: Save YAML back to GridFS ===
    file_id, identifier = fp.add_file(
        tmp_yml_path,
        identifier=str(uuid4()),
        metadata={
            "firework_name": "convert_mets_to_yml",
            "source_mets_gfs_id": gfs_id,
            "accession_number": accession_number,
        },
    )

    print(f"Wrote YAML to GridFS with ID: {file_id}")

    return FWAction(
        update_spec={"yml_gfs_id": file_id, "yml_identifier": identifier},
        stored_data={"yml_gfs_id": file_id},
    )


def image_conversion_task(*args):
    identifiers = args[0]
    accession_number = args[1]
    identifiers_out = []
    i = 1
    for file_id in tqdm(identifiers, desc="Processing images"):
        # Get raw file bytes
        file_contents, doc = fp.get_file(file_id)

        # Convert bytes -> numpy array -> OpenCV image
        nparr = np.frombuffer(file_contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        rotated = rotate(img, angle, (255, 255, 255))

        # Check if image was loaded successfully
        if img is None:
            print(f"Failed to load image: {file_id}")
            continue

        # Process the image
        processed_img = process_historical_document(rotated)
        # Now you can save or use the processed image
        # For example, encode back to bytes if needed:

        success, encoded_img = cv2.imencode(".png", processed_img)
        if not success:
            raise ValueError("Failed to encode image")
        suffix = f"{accession_number}_{i:010d}.png"
        # Save processed image to a temporary file
        tmp_dir = tempfile.gettempdir()
        filename = os.path.join(tmp_dir, suffix)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        s3.upload_fileobj(
            encoded_img.tobytes(),
            "environmental-impact-statements-data",
            f"{accession_number}/PNG/{suffix}",
        )
        with open(filename, "wb") as tmp_file:
            tmp_file.write(encoded_img.tobytes())
            tmp_path = tmp_file.name
            print(tmp_path)

        # Add file to your file manager / database
        file_id, identifier = fp.add_file(
            tmp_path,
            identifier=str(uuid4()),
            metadata={
                "firework_name": "image_conversion",
                "accession_number": accession_number,
            },
        )  # adjust this to your API
        identifiers_out.append(identifier)
        i += 1
    return FWAction(update_spec={"converted_images": identifiers_out})


def image_to_pdf(*args):
    """
    Combines JPEG images from GridFS into a single PDF, saves it to a temp file,
    and adds it to GridFS.
    args[0] = list of GridFS file IDs
    args[1] = optional barcode_dir (not used here)
    """
    identifiers = args[0]
    accession_number = args[1]
    images = []

    for file_id in identifiers:
        # Get raw file bytes
        file_contents, doc = fp.get_file(file_id)

        # Load image from bytes
        img = Image.open(io.BytesIO(file_contents))
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    if not images:
        raise ValueError("No images were retrieved from GridFS")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        temp_pdf_path = tmp_file.name
        # Save images to PDF
        images[0].save(
            temp_pdf_path, save_all=True, append_images=images[1:], format="PDF"
        )
        file_id, identifier = fp.add_file(
            temp_pdf_path,
            identifier=str(uuid4()),
            metadata={
                "firework_name": "image_to_pdf",
                "accession_number": accession_number,
            },
        )
        print(file_id, identifier)

    return FWAction(
        update_spec={"PDF_id": [identifier], "pdf_gfs_id": file_id},
        stored_data={"pdf_gfs_id": file_id},
    )


def marker_on_image(*args):
    """
    Adds a marker to each page of a PDF.
    args[0] = GridFS file ID of the PDF (may be nested)
    args[1] = optional barcode_dir (not used here)
    """
    from PIL import Image
    from io import BytesIO

    image_id = args[0][0] if isinstance(args[0], (list, tuple)) else args[0]
    accession_number = args[1]
    file_contents, doc = fp.get_file(image_id)
    print(doc)
    logger.debug(f"Type of file content: {type(file_contents)}")
    logger.info("Successfully called doc")

    config = {
        "output_format": "json",
        "paginate_output": True,
    }

    config_parser = ConfigParser(config)
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    predictions = recognition_predictor(
        [Image.open(BytesIO(file_contents))], det_predictor=detection_predictor
    )
    logger.debug(f"Type of OCR Result: {type(predictions[0])}")
    return FWAction(update_spec={"marker_output": predictions}, stored_data={})


def marker_on_pdf(*args):
    """
    Adds a marker to each page of a PDF.
    args[0] = GridFS file ID of the PDF (may be nested)
    args[1] = optional barcode_dir (not used here)
    """
    import base64
    import binascii
    import logging
    import os
    import re
    from pathlib import Path

    pdf_id = args[0][0] if isinstance(args[0], (list, tuple)) else args[0]
    accession_number = args[1]
    file_contents, doc = fp.get_file(pdf_id)
    print(doc)
    # Normalize to raw bytes
    if hasattr(file_contents, "read"):
        try:
            file_contents = file_contents.read()
        except Exception as e:
            raise RuntimeError(f"Failed reading GridFS file for {pdf_id}: {e}")

    # If it is a str, maybe it's base64 or plain text (invalid)
    if isinstance(file_contents, str):
        if file_contents.startswith("%PDF"):
            file_contents = file_contents.encode("utf-8")
        else:
            try:
                file_contents = base64.b64decode(file_contents, validate=True)
            except binascii.Error:
                raise ValueError(
                    "File content is a string and not valid base64 nor a PDF header"
                )

    if not isinstance(file_contents, (bytes, bytearray)):
        raise TypeError(f"Unexpected file_contents type: {type(file_contents)}")

    # Dump raw retrieved bytes for inspection (even if empty / invalid)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(pdf_id))[:60]
    if file_contents.startswith(b"%PDF"):
        debug_name = f"retrieved_{safe_id}.pdf"
    else:
        debug_name = f"retrieved_{safe_id}.bin"
    try:
        with open(debug_name, "wb") as dbg:
            dbg.write(file_contents)
        logging.info(
            f"Wrote raw retrieved file to {os.path.abspath(debug_name)} (size={len(file_contents)})"
        )
    except Exception as e:
        logging.warning(f"Failed writing debug file {debug_name}: {e}")

    # Basic PDF sanity checks
    if not file_contents.startswith(b"%PDF"):
        preview = file_contents[:32]
        raise ValueError(
            f"Retrieved data does not look like a PDF. Starts with: {preview}"
        )

    if b"%%EOF" not in file_contents[-2048:]:
        logging.warning("PDF missing trailing %%EOF marker (may be truncated)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        temp_pdf_path = tmp_file.name
        tmp_file.write(file_contents)

    if Path(temp_pdf_path).stat().st_size == 0:
        raise ValueError("Temporary PDF file is empty after write")
    # Save it to current dir for inspection test
    with open(f"debug_{safe_id}.pdf", "wb") as dbg:
        dbg.write(file_contents)

    # Run Marker on PDF via Python API
    config = {
        "output_format": "json",
        "paginate_output": True,
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    try:
        rendered: JSONOutput
        rendered = converter(temp_pdf_path)
        # Persist rendered JSON via a secure temp file (avoid cluttering CWD)
        s3.upload_fileobj(
            rendered,
            "environmental-impact-statements-data",
            f"{accession_number}/schema.json",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp_json:
            json_path = tmp_json.name
            tmp_json.write(rendered.model_dump_json(indent=2))

        try:
            file_id, id = fp.add_file(
                json_path,
                identifier=str(uuid4),
                metadata={
                    "firework_name": "marker_on_pdf",
                    "accession_number": accession_number,
                },
            )
            print(file_id, id)
        finally:
            try:
                os.unlink(json_path)
            except OSError:
                pass
    except Exception as e:
        raise RuntimeError(
            f"Marker/PDFium failed on {pdf_id} (temp: {temp_pdf_path}): {e}"
        )

    return FWAction(update_spec={"marker_output": rendered}, stored_data={})


def get_requested_file_gfs_id_yaml(target_id):
    client = MongoClient(url)

    db = client["fireworks"]
    filepad_col = db["filepad"]
    result = filepad_col.find_one(
        {
            "metadata.accession_number": target_id,
            "metadata.firework_name": "convert_mets_to_yml",
        }
    )
    print(result)
    gfs_id = result.get("gfs_id", {})
    return gfs_id


def get_requested_file_gfs_id_json(target_id):
    client = MongoClient(url)

    db = client["fireworks"]
    filepad_col = db["filepad"]
    result = filepad_col.find_one(
        {
            "metadata.accession_number": target_id,
            "metadata.firework_name": "marker_on_pdf",
        }
    )
    if isinstance(result, dict):
        print(result)
        gfs_id = result.get("gfs_id")
        return gfs_id
    else:
        logger.info("Could not find JSON Documents!")


def get_requested_file_gfs_id_pdf(accession_number: str) -> str | None:
    """
    `get_requested_file_gfs_id_pdf`
    """

    client = MongoClient(url)

    db = client["fireworks"]
    filepad_col = db["filepad"]
    result = filepad_col.find_one(
        {
            "metadata.accession_number": accession_number,
            "metadata.firework_name": "image_to_pdf",
        }
    )
    return result.get("gfs_id")


def create_ingest_sheet():
    import pandas as pd


def get_requested_file_gfs_id_pngs(target_id):
    client = MongoClient(url)

    db = client["fireworks"]
    filepad_col = db["filepad"]
    result = filepad_col.find(
        {
            "metadata.accession_number": target_id,
            "metadata.firework_name": "image_conversion",
        }
    )
    print(result)
    gfs_ids = []
    for doc in result:
        gfs_id = doc.get("gfs_id")
        gfs_ids.append((gfs_id, doc.get("original_file_name", "")))
    return gfs_ids


def upload_file_to_s3(file_contents, doc, s3_path):
    # Ensure doc and file_contents are not None before proceeding
    if doc is not None and file_contents is not None:
        output_path = s3_path
        with open(output_path, "wb") as f:
            f.write(file_contents)
    else:
        print("File or document not found.")


def dump_data_to_s3(accession_number: str):
    client = MongoClient(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
    db = client["fireworks"]
    coll = db["filepad"]
    gfs_id = get_requested_file_gfs_id_pdf(accession_number)

    return FWAction(update_spec=None, stored_data=None)
