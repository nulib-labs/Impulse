from fireworks.utilities.filepad import FilePad
from typing import Type
import os
import certifi
from pymongo import MongoClient
import argparse
from loguru import logger
import json
from bs4 import BeautifulSoup

JSONLike = dict[str, dict[str, str]]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--accession_number",
    "-id",
    help="The barcode identifier for what you wish to download.",
)
parser.add_argument(
    "--output_file", "-o", help="A file path to which the file will be saved."
)

args = parser.parse_args()

accession_number: str = args.accession_number
output_file: str = args.output_file

url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
OUTPUT_DIR = f"downloaded_files/{accession_number}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_mongo_client_kwargs():
    return {
        "tls": True,
        "tlsCAFile": certifi.where(),
    }


def get_requested_file_gfs_id_pdf(accession_number: str) -> str | None:
    """
    `get_requested_file_gfs_id_pdf`
    """

    try:
        client = MongoClient(url)
        db = client["fireworks"]
    except Exception as e:
        logger.error(f"Failed to connect to MongoClient.\n{e}")
        raise e
    filepad_col = db["filepad"]
    result: dict | None
    result = filepad_col.find_one(
        {
            "metadata.accession_number": accession_number,
            "metadata.firework_name": "image_to_pdf",
        }
    )
    if isinstance(result, dict):
        return result.get("gfs_id")
    else:
        return None


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


def save_file(file_contents, doc, output_file_name):
    # Ensure doc and file_contents are not None before proceeding
    if doc is not None and file_contents is not None:
        output_path = os.path.join(OUTPUT_DIR, output_file_name)
        with open(output_path, "wb") as f:
            f.write(file_contents)
    else:
        print("File or document not found.")


def find_all_html(data: JSONLike) -> list[str]:
    results = []

    for key, value in data.items():
        if key == "html":
            results.append(value)

    return results


def create_txt_files(data: JSONLike):
    logger.info("Now creating text files.")
    results: list[str] = find_all_html(data)
    logger.info(f"Length of results: {len(results)}")
    cleantext = BeautifulSoup(text, "lxml").text
    with open("Cleaned.txt", "w") as f:
        f.write(cleantext)
    pass


def get_filecontents_and_doc(gfs_id):
    conn_str: str
    conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))

    DB_NAME = "fireworks"
    FILEPAD_COLLECTION = "filepad"
    GRIDFS_COLLECTION = "filepad_gfs"

    fp = FilePad(
        host=conn_str + "/fireworks?",
        port=27017,
        uri_mode=True,
        mongoclient_kwargs=get_mongo_client_kwargs(),
    )

    file_contents, doc = fp.get_file_by_id(gfs_id)
    return file_contents, doc


if __name__ == "__main__":
    logger.info(f"Looking for files with accession number {accession_number}")
    gfs_id = get_requested_file_gfs_id_pdf(accession_number)
    try:
        yaml_file = get_requested_file_gfs_id_yaml(accession_number)
        file_contents, doc = get_filecontents_and_doc(yaml_file)
        save_file(file_contents, doc, "meta.yaml")
        if file_contents is not None and doc is not None:
            save_file(file_contents, doc, output_file if output_file else "output.pdf")
    except:
        print("No YAML file found!")
    gfs_ids = get_requested_file_gfs_id_pngs(accession_number)
    for gfs_id, original_file_name in gfs_ids:
        file_contents, doc = get_filecontents_and_doc(gfs_id)
        save_file(file_contents, doc, original_file_name)

    gfs_id = get_requested_file_gfs_id_json(accession_number)
    logger.info(f"Jsons GFS_ID: {gfs_id}")
    file_contents: bytes | None
    file_contents, doc = get_filecontents_and_doc(gfs_id)
    logger.info(f"Type of file_contents: {type(file_contents)}")
    save_file(file_contents, doc, f"{accession_number}.json")
    logger.info(
        f"Head of file_contents: {file_contents[:50] if file_contents is not None else 'file_contents has None value'}"
    )

    file_contents_str: str | TypeError
    file_contents_str = (
        file_contents.decode("utf-8") if file_contents is not None else TypeError()
    )

    logger.info(
        f"Head of file_contents_str: {file_contents_str[:50] if file_contents_str is str else TypeError}"
    )

    pass
