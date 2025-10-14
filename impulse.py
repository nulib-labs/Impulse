import argparse
import os
import fireworks
from pymongo import MongoClient
from fireworks.utilities.filepad import FilePad

conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
lpad = FilePad(host=(conn_str + "/fireworks?"), uri_mode=True)


def save_file(file_contents, doc, output_file_name):
    # Ensure doc and file_contents are not None before proceeding
    if doc is not None and file_contents is not None:
        filename = doc.get("original_file_name", "")
        with open(output_file_name, "wb") as f:
            f.write(file_contents)
    else:
        print("File or document not found.")


def download_work(accession_number):
    """
    Gets all entries given an accession number.
    """
    url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
    client = MongoClient(url)
    db = client["fireworks"]
    coll = db["filepad"]
    png_files = coll.find(
        {
            "metadata.firework_name": "image_conversion",
            "metadata.accession_number": str(accession_number),
        }
    )

    pdf_file = coll.find_one(
        {
            "metadata.firework_name": "image_to_pdf",
            "metadata.accession_number": str(accession_number),
        }
    )
    base_dir = os.path.join("output", accession_number)
    processed_dir = os.path.join(base_dir, "processed_images")
    os.makedirs(processed_dir, exist_ok=True)

    for doc in png_files:
        gfs_id = doc.get("gfs_id")
        filename_doc = coll.find_one({"gfs_id": gfs_id})
        filename = filename_doc.get("original_file_name") if filename_doc else None
        if not filename:
            print(f"Skipping document with gfs_id {gfs_id}: no filename found.")
            continue
        file_contents, doc = lpad.get_file_by_id(gfs_id)
        save_file(file_contents, doc, os.path.join(processed_dir, str(filename)))

    pdf_doc = pdf_file
    gfs_id = pdf_doc.get("gfs_id")
    filename_doc = coll.find_one({"gfs_id": gfs_id})
    filename = filename_doc.get("original_file_name") if filename_doc else None
    file_contents, doc = lpad.get_file_by_id(gfs_id)
    pdf_dir = os.path.join("output", accession_number, "PDF")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{accession_number}.pdf")
    save_file(file_contents, pdf_doc, pdf_path)
    pass


def main():
    parser = argparse.ArgumentParser(
        prog="The Impulse CLI",
        description="This program allows digitization team to download files and work with them.",
    )
    parser.add_argument("--accession_number", "-a", help="The accession number")
    args = parser.parse_args()
    accession_number = args.accession_number

    download_work(accession_number)


main()
