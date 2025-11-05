import os
import glob
import json
from pymongo import MongoClient
from bs4 import BeautifulSoup
from loguru import logger
from PIL import Image
import argparse

parser = argparse.ArgumentParser(prog="HathiTrust Packager")
parser.add_argument("dirname", type=str)
parser.add_argument("--accession_number", "-a")
args = parser.parse_args()
directory = args.dirname
accession_number = args.accession_number
# --- Setup ---
client = MongoClient(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
db = client["fireworks"]
coll = db["filepad"]


# --- Helper function ---
def find_all_html(data):
    """Recursively find all 'html' values in nested dicts/lists."""
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "html":
                results.append(value)
            else:
                results.extend(find_all_html(value))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_all_html(item))
    return results


def main():
    json_files = glob.glob(f"{directory}/**/*.json", recursive=True)
    logger.info(f"JSON files found: {json_files}")

    for i, path in enumerate(json_files):
        input_dir_pngs = glob.glob(f"{directory}/{accession_number}/*.png")
        accession_number_dir = os.path.join(os.getcwd(), accession_number)
        os.makedirs(accession_number_dir, exist_ok=True)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load data from json file:\n{e}")
            data = {}

        j = 1
        for k, page in enumerate(data["children"]):
            html_fragments = find_all_html(data["children"][k])
            combined_html = " ".join(html_fragments)
            cleantext = BeautifulSoup(combined_html, "lxml").get_text(
                separator=" ", strip=True
            )
            text_dir = path.replace(".json", "") + "/TEXT/"
            os.makedirs((accession_number_dir + "/TEXT/"), exist_ok=True)
            j_str = str(j)
            with open(
                accession_number_dir + f"/TEXT/{accession_number}_{j:010d}.TXT",
                "w+",
            ) as f:
                f.write(cleantext)
            j += 1
        os.makedirs(accession_number_dir + "/JP2/", exist_ok=True)
        j = 1
        for x in input_dir_pngs:
            img = Image.open(x)
            img.save(
                (accession_number_dir + f"/JP2/{accession_number}_{j:010d}.JP2"),
                format="JPEG2000",
            )
            j += 1


if __name__ == "__main__":
    main()
