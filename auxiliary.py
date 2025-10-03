from uuid import uuid4
from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
import certifi
from fireworks.core.firework import FWAction, Firework
from fireworks.user_objects.firetasks.script_task import PyTask
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

conn_str: str
conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))


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
    host=conn_str + "/fireworks?",
    port=27017,
    uri_mode=True,
    database="fireworks",
    mongoclient_kwargs=get_mongo_client_kwargs(),
)


def printurn(*args):
    result = []
    for arg in args:
        if isinstance(arg, list) and len(arg) == 1:
            result.append(arg[0])
        else:
            result.append(arg)
    if len(result) == 1:
        result = result[0]
    print(result)
    return result


def image_conversion_task(*args):
    w = woolworm.Woolworm()
    identifiers = args[0]
    barcode_dir = args[1]
    identifiers_out = []
    i = 1
    for file_id in tqdm(identifiers, desc="Processing images"):
        # Get raw file bytes
        file_contents, doc = fp.get_file(file_id)
        # Convert bytes -> numpy array -> OpenCV image
        nparr = np.frombuffer(file_contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process image
        img, _ = w.binarize_or_gray(img)
        img = w.deskew_with_hough(img)
        img = w.remove_borders(img)

        # Encode as PNG in memory
        success, encoded_img = cv2.imencode(".png", img)
        if not success:
            raise ValueError("Failed to encode image")
        suffix = f"{barcode_dir}_{i:010d}.png"
        # Save processed image to a temporary file
        tmp_dir = tempfile.gettempdir()
        filename = os.path.join(tmp_dir, suffix)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as tmp_file:
            tmp_file.write(encoded_img.tobytes())
            tmp_path = tmp_file.name
            print(tmp_path)

        # Add file to your file manager / database
        file_id, identifier = fp.add_file(
            tmp_path, identifier=str(uuid4())
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
        file_id, identifier = fp.add_file(temp_pdf_path, identifier=str(uuid4()))

    return FWAction(update_spec={"PDF_id": [identifier]})


def marker_on_pdf(*args):
    """
    Adds a marker to each page of a PDF.
    args[0] = GridFS file ID of the PDF (may be nested)
    args[1] = optional barcode_dir (not used here)
    """
    import base64, binascii, logging, os, re
    from pathlib import Path

    pdf_id = args[0][0] if isinstance(args[0], (list, tuple)) else args[0]

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

    # Run Marker on PDF via Python API
    config = {"output_format": "json"}
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp_json:
            json_path = tmp_json.name
            tmp_json.write(rendered.model_dump_json(indent=2))

        try:
            fp.add_file(json_path, identifier=f"marked_{safe_id}.json")
        finally:
            try:
                os.unlink(json_path)
            except OSError:
                pass
    except Exception as e:
        raise RuntimeError(
            f"Marker/PDFium failed on {pdf_id} (temp: {temp_pdf_path}): {e}"
        )

    return FWAction(update_spec={"marker_output": rendered})
