from uuid import uuid4
from fireworks.fw_config import os
import certifi
from fireworks.core.firework import FWAction
from marker.renderers.json import JSONOutput
import cv2
import numpy as np
import tempfile
import io
from PIL import Image, UnidentifiedImageError
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
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
from my_pads import fp, lp
import pandas as pd

# Global vars
conn_str: str
conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
fp = fp
lp = lp

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("MEADOW_PROD_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("MEADOW_PROD_SECRET_ACCESS_KEY"),
)


def make_ingest_sheet(*args):
    from io import StringIO, BytesIO

    logger.info(f"Ingest sheet args: {args}")
    filenames: str = args[0]
    accession_number = args[1]
    file_accession_numbers = []
    for f in filenames:
        file_accession_numbers.append(f.split("/")[-1].split(".")[0])

    df = pd.DataFrame(
        {
            "work_type": ["IMAGE" for i in filenames],
            "work_accession_number": [accession_number for i in filenames],
            "file_accession_number": [str(Path(f)).stem for f in filenames],
            "filename": [
                "/".join([accession_number, "SOURCE", "jpg", f.replace(".jp2", ".jpg")])
                for f in filenames
            ],
            "description": [f.replace(".jp2", ".jpg") for f in filenames],
            "role": ["A" for i in filenames],
            "label": [i for i, f in enumerate(filenames)],
            "work_image": ["" for i in filenames],
            "structure": [
                "/".join([accession_number, "TXT", f.replace(".jpg", ".txt")])
                for f in filenames
            ],
        }
    )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Reset buffer position to beginning
    csv_buffer.seek(0)

    # Convert StringIO to BytesIO for upload_fileobj
    bytes_buffer = BytesIO(csv_buffer.getvalue().encode())

    ingest_key = "/".join(["p0491p1074eis-1766005955", accession_number, "ingest.csv"])

    s3.upload_fileobj(bytes_buffer, "meadow-p-ingest", ingest_key)
    pass


def rotate(
    image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Rotates an OpenCV image according to an angle,
    fills background with a color as necessary.
    """

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
    # proportion of dark pixels
    foreground_ratio = np.sum(binary == 0) / binary.size
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


def spacy_experiment(*args):
    """
    EXPERIMENTAL.

    This function is experimental and subject to change or removal
    without notice.

    FireWorks task function: Pull the content of a .txt file from S3,
    run Spacy NER function on it, upload the NER results back to S3
    args[0] = s3_key of the METS XML file in S3
    args[1] = filename
    args[2] = accession_number
    """

    import spacy  # import spacy
    import json

    s3_impulse = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
    )

    s3_key: str = args[0]
    filename: str = args[1]
    accession_number: str = args[2]

    print(s3_key)
    print(filename)
    print(accession_number)

    # Build out the output file names
    output_filename = filename.replace(".txt", ".json")
    ner_s3_key = "/".join([accession_number, "SPACY_NER", output_filename])

    response = s3_impulse.get_object(Bucket="nu-impulse-production", Key=s3_key)
    data: bytes = response["Body"].read()
    print(type(data))
    text = data.decode("utf-8").replace("\n", " ")

    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)
    # Save NER as JSON
    ner_dict = {
        "text": doc.text,
        "entities": [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            }
            for ent in doc.ents
        ],
    }

    s3_impulse.put_object(
        Bucket="nu-impulse-production",
        Key=ner_s3_key,
        Body=json.dumps(ner_dict, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

    return True


def convert_mets_to_yml(*args):
    """
    FireWorks task function: Convert a METS XML file from S3 into a YAML file
    following HathiTrust ingest specs, and save the YAML back to S3.
    args[0] = s3_key of the METS XML file in S3
    args[1] = filename
    args[2] = accession_number
    """

    s3_key = args[0]
    filename = args[1]
    accession_number = args[2]

    s3_impulse = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
    )

    output_filename = filename.replace(".xml", ".yaml")
    logger.info(f"Value of output filename: {output_filename}")
    response = s3_impulse.get_object(Bucket="nu-impulse-production", Key=s3_key)
    data = response["Body"].read()

    # === Step 2: Parse the XML directly from memory ===
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.ElementTree(ET.fromstring(data, parser))
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
        page_filename = find_filename_by_file_id(file_id)
        if not page_filename:
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
                line = f'{page_filename}: {{ label: "{label}" }}'
            elif parent_label == "Front Matter" and orderlabel:
                line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'
            elif parent_label == "Title":
                label = "TITLE"
                line = (
                    f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                        label
                    }" }}'
                    if orderlabel
                    else f'{page_filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Contents":
                label = "TABLE_OF_CONTENTS"
                line = (
                    f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                        label
                    }" }}'
                    if orderlabel
                    else f'{page_filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Preface":
                label = "PREFACE"
                line = (
                    f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                        label
                    }" }}'
                    if orderlabel
                    else f'{page_filename}: {{ label: "{label}" }}'
                )
            elif parent_label.startswith("Chapter") or parent_label == "Appendix":
                label = "CHAPTER_START"
                line = (
                    f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                        label
                    }" }}'
                    if orderlabel
                    else f'{page_filename}: {{ label: "{label}" }}'
                )
            elif parent_label in ("Notes", "Bibliography"):
                label = "REFERENCES"
                line = (
                    f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                        label
                    }" }}'
                    if orderlabel
                    else f'{page_filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Index":
                label = "INDEX"
                line = (
                    f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                        label
                    }" }}'
                    if orderlabel
                    else f'{page_filename}: {{ label: "{label}" }}'
                )
            elif parent_label == "Cover" and parent_type == "cover":
                label = "BACK_COVER"
                line = f'{page_filename}: {{ label: "{label}" }}'
        else:
            if orderlabel:
                line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'

        if line:
            yaml_lines.append("    " + line)

    # === Step 5: Write YAML to S3 ===
    yaml_content = "\n".join(yaml_lines)

    s3_output_key = (
        s3_key.rsplit("/", 1)[0] + "/mets.yaml" if "/" in s3_key else "mets.yaml"
    )

    try:
        s3_impulse.put_object(
            Bucket="nu-impulse-production",
            Key=s3_output_key,
            Body=yaml_content.encode("utf-8"),
            ContentType="application/x-yaml",
        )
        logger.info(f"Successfully wrote YAML to S3: {s3_output_key}")
    except Exception as e:
        logger.error(f"Failed to write YAML to S3: {e}")
        raise

    return FWAction(
        update_spec={"yml_s3_key": s3_output_key, "yml_identifier": accession_number},
        stored_data={"yml_s3_key": s3_output_key},
    )


def image_conversion_task(*args):
    logger.info(f"Value of args: {args}")
    logger.info(f"Value of args[0]: {args[0]}")
    logger.info(f"Value of args[1]: {args[1]}")
    gfs_id = args[0][0][1]
    file_name = args[0][0]
    logger.info(f"Now running on image_id: {gfs_id, file_name}")
    accession_number = args[1]
    file_contents, doc = fp.get_file(gfs_id)
    logger.info(
        f"Running `image_conversion_task` on {file_name}, with accession number {
            accession_number
        }"
    )

    # Get raw file bytes
    file_contents, doc = fp.get_file(gfs_id)

    # Convert bytes -> numpy array -> OpenCV image
    nparr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = rotate(img, angle, (255, 255, 255))

    # Process the image
    processed_img = process_historical_document(rotated)
    # Now you can save or use the processed image
    # For example, encode back to bytes if needed:

    success, encoded_img = cv2.imencode(".png", processed_img)
    if not success:
        raise ValueError("Failed to encode image")
    suffix = f"{accession_number}_{file_name}.png"
    # Save processed image to a temporary file
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, suffix)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info(f"Now uploading encoded image {filename} to S3. DUMMY")
    # s3.upload_fileobj(
    #     io.BytesIO(encoded_img.tobytes()),
    #     "environmental-impact-statements-data",
    #     f"{accession_number}/PNG/{suffix}",
    # )
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

    return FWAction(update_spec={"converted_images": identifier})


def image_to_pdf(*args):
    """
    Combines JPEG images from GridFS into a single PDF, saves it to a temp file,
    and adds it to GridFS.
    args[0] = list of S3 file keys
    args[1] = optional accession_number (not used here)
    """
    s3_keys = args[0]
    accession_number = args[1]
    images = []

    s3_impulse = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
    )

    for s3_key in s3_keys:
        logger.info(f"Now running on {s3_key}")

        response = s3_impulse.get_object(Bucket="nu-impulse-production", Key=s3_key)

        file_contents = response["Body"].read()

        if not file_contents or len(file_contents) < 1024:
            logger.warning(f"Skipping {s3_key}: empty or too small")
            continue

        try:
            with Image.open(io.BytesIO(file_contents)) as img:
                img.verify()  # validate before loading
            img = Image.open(io.BytesIO(file_contents))  # reopen after verify

            if img.mode != "RGB":
                img = img.convert("RGB")

            images.append(img)

        except UnidentifiedImageError:
            logger.error(f"Not an image or unsupported format: {s3_key}")
            continue
        except Exception as e:
            logger.exception(f"Failed processing {s3_key}: {e}")
            continue

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        temp_pdf_path = tmp_file.name
        # Save images to PDF
        images[0].save(
            temp_pdf_path, save_all=True, append_images=images[1:], format="PDF"
        )
        s3_impulse.upload_file(
            temp_pdf_path,
            "nu-impulse-production",
            "/".join([accession_number, "main.pdf"]),
        )
        s3.upload_file(
            temp_pdf_path,
            "meadow-p-ingest",
            "/".join(["p0491p1074eis-1766005955", accession_number, "main.pdf"]),
        )

    return None


def surya_on_image(*args):
    """
    Adds a marker to each page of a PDF.
    args[0] = S3 Key of the Image
    """
    from PIL import Image
    from io import BytesIO
    import json

    s3_key = args[0]
    filename: str = args[1]
    accession_number = args[2]

    s3_impulse = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
    )

    output_filename = filename.replace(".jp2", ".txt")
    confidence_output_filename = filename.replace(".jp2", ".json")

    logger.info(f"Value of output filename: {output_filename}")
    response = s3.get_object(Bucket="meadow-p-ingest", Key=s3_key)
    data = response["Body"].read()

    # Make OCR Predictions
    logger.info("Making Predictions")
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    predictions = recognition_predictor(
        [Image.open(BytesIO(data))], det_predictor=detection_predictor
    )

    # Extract text from json
    logger.info("Extracting text.")
    text_key = "/".join(
        ["p0491p1074eis-1766005955", accession_number, "TXT", output_filename]
    )
    impulse_text_key = "/".join(
        ["p0491p1074eis-1766005955", accession_number, "TXT", output_filename]
    )
    confidence_key = "/".join(
        [accession_number, "CONFIDENCES", output_filename.replace(".txt", ".json")]
    )
    logger.info(f"Saving text to {text_key}")

    text_lines = []
    predictions_data = []

    for prediction in predictions:
        data = prediction.model_dump()
        predictions_data.append(data)

        for line in data["text_lines"]:
            text_lines.append(line["text"])

    text_bytes = "\n".join(text_lines).encode("utf-8")

    predictions_bytes = json.dumps(
        predictions_data,
        ensure_ascii=False,  # preserve unicode text
        indent=2,  # optional, for readability
    ).encode("utf-8")

    s3.put_object(
        Body=text_bytes,
        Bucket="meadow-p-ingest",
        Key=text_key,
        ContentType="text/plain; charset=utf-8",
    )

    s3_impulse.put_object(
        Body=text_bytes,
        Bucket="nu-impulse-production",
        Key=impulse_text_key,
        ContentType="application/json",
    )
    s3_impulse.put_object(
        Body=predictions_bytes,
        Bucket="nu-impulse-production",
        Key=confidence_key,
        ContentType="application/json",
    )

    return [text_bytes]
