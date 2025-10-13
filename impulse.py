import argparse
import os
import fireworks
from pymongo import MongoClient


def download_work(accession_number):
    """
    Gets all entries given an accession number.
    """
    url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
    client = MongoClient(url)
    db = client["fireworks"]
    coll = db["filepad"]
    pass


def main():
    parser = argparse.ArgumentParser(
        prog="The Impulse CLI",
        description="This program allows digitization team to download files and work with them.",
    )
    parser.add_argument(
        "--download_work",
        "-d",
        help="This command downloads all output files from a work.",
    )

    pass
