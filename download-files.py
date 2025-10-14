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

OUTPUT_DIR = f"./downloaded_files/{target_id}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_mongo_client_kwargs():
    return {
        "tls": True,
        "tlsCAFile": certifi.where(),
    }


def get_requested_file_gfs_id_pdf(target_id):
    client = MongoClient(url)

    db = client["fireworks"]
    filepad_col = db["filepad"]
    result = filepad_col.find_one(
        {
            "metadata.accession_number": target_id,
            "metadata.firework_name": "image_to_pdf",
        }
    )
    print(result)
    gfs_id = result.get("gfs_id")
    return gfs_id


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


gfs_id = get_requested_file_gfs_id_pdf(target_id)
file_contents, doc = get_filecontents_and_doc(gfs_id)
if file_contents is not None and doc is not None:
    save_file(file_contents, doc, output_file if output_file else "output.pdf")
gfs_ids = get_requested_file_gfs_id_pngs(target_id)
for gfs_id, original_file_name in gfs_ids:
    file_contents, doc = get_filecontents_and_doc(gfs_id)
    save_file(file_contents, doc, original_file_name)
