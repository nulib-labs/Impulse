from fireworks.utilities.filepad import FilePad
import os
import certifi
from pymongo import MongoClient
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target_id", "-id", help="The barcode identifier for what you wish to download."
)
parser.add_argument(
    "--output_file", "-o", help="A file path to which the file will be saved."
)
args = parser.parse_args()

target_id = args.target_id
output_file = args.output_file

url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")

OUTPUT_DIR = "./downloaded_files"


def get_mongo_client_kwargs():
    return {
        "tls": True,
        "tlsCAFile": certifi.where(),
    }


def get_requested_file_gfs_id(target_id):
    client = MongoClient(url)

    db = client["fireworks"]
    filepad_col = db["filepad"]
    result = filepad_col.find_one(
        {"metadata.identifier": target_id, "metadata.firework_name": "image_to_pdf"}
    )
    gfs_id = result.get("gfs_id")
    return gfs_id


def save_file(file_contents, doc, output_file_name):
    # Ensure doc and file_contents are not None before proceeding
    if doc is not None and file_contents is not None:
        filename = doc.get("original_file_name", "")
        with open(output_file_name, "wb") as f:
            f.write(file_contents)
    else:
        print("File or document not found.")


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
        database="fireworks",
        mongoclient_kwargs=get_mongo_client_kwargs(),
    )

    file_contents, doc = fp.get_file_by_id(gfs_id)
    return file_contents, doc


gfs_id = get_requested_file_gfs_id(target_id)
file_contents, doc = get_filecontents_and_doc(gfs_id)
save_file(file_contents, doc, output_file)
