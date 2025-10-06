from pymongo import MongoClient
from fireworks.utilities.filepad import FilePad
import gridfs
import os
from bson import ObjectId
import gzip
import io
import certifi


def get_mongo_client_kwargs():
    return {
        "tls": True,
        "tlsCAFile": certifi.where(),
    }


conn_str: str
conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
# === CONFIGURATION ===
MONGO_URI = os.getenv(
    "MONGODB_OCR_DEVELOPMENT_CONN_STRING"
)  # or your remote MongoDB URI
DB_NAME = "fireworks"
FILEPAD_COLLECTION = "filepad"
GRIDFS_COLLECTION = "filepad_gfs"
OUTPUT_DIR = "./downloaded_files"

fp = FilePad(
    host=conn_str + "/fireworks?",
    port=27017,
    uri_mode=True,
    database="fireworks",
    mongoclient_kwargs=get_mongo_client_kwargs(),
)
# Save
file_contents, doc = fp.get_file_by_id("68e405af7380cf221007b399")

# Ensure doc and file_contents are not None before proceeding
if doc is not None and file_contents is not None:
    filename = doc.get("original_file_name", "")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
        f.write(file_contents)
    print(file_contents)
else:
    print("File or document not found.")
