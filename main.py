from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask
import glob
import argparse
from tqdm import tqdm
from fabric import Connection
from loguru import logger
from pathlib import Path
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("MEADOW_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("MEADOW_SECRET_ACCESS_KEY")
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    "-i",
    help="The input path. If the input path is a directory, iterates over all files inside. Pass --recursive option to loop through all child directories.",
)
parser.add_argument(
    "--recursive",
    "-r",
    help="If this argument is passed, adds files recursively to the workflow.",
    action="store_true",
)

parser.add_argument(
    "--accession_number",
    "-a",
    help="The identifier for which the files should be linked. This should be a barcode.",
)

parser.add_argument(
    "--fw_name",
    "-n",
    help="A custom name of the workflow. If no argument is passed, defaults to value of --identifier",
    required=False,
)

parser.add_argument(
    "--reset",
    help="Forces reset of all data. If you do not have admin access, this will fail.",
    required=False,
    action="store_true",
)

args = parser.parse_args()

input_path = args.input
accession_number = args.accession_number
is_recursive = args.recursive
fw_name = args.fw_name
do_reset = args.reset


conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
lp = LaunchPad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    name="fireworks",
)
fp = FilePad(
    host=conn_str,
    port=27017,
    uri_mode=True,
)
if do_reset:
    lp.reset("2025-10-08", require_password=False)
    fp.reset()

suffixes = ["jpg", "png", "jpeg", "tiff", "jp2", "xml"]

files = []

if is_recursive:
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(input_path, "**",
                      f"*.{suffix}"), recursive=True)
        )
    files = sorted(files)
    print(files)
    pass

else:
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(input_path, "**",
                      f"*.{suffix}"), recursive=True)
        )
    print(files)
    specs = []
    for f in tqdm(sorted(files), desc="Uploading files..."):
        f = Path(f)
        logger.info(f"Name of file: {f.name}")
        logger.info(f"Value of f: {f}")
        new_identifier = str(accession_number) + "_" + str(f.name).zfill(10)
        logger.info(f"Value of new identifier: {new_identifier}")
        print(new_identifier)
        logger.info(f"Now attempting to upload {f} to S3")
        key = "/".join([accession_number, "SOURCE", f.suffix[1:], f.name])

        logger.debug(f"Value of key: {key}")
        logger.debug(f"Value of accession number: {accession_number}")
        logger.debug(f"Value of f: {f}")

        specs.append({"source_s3_key": key,
                      "file_name": f.name,
                      "accession_number": accession_number},)
        with open(f, 'rb') as data:
            s3.upload_fileobj(data,
                              'meadow-s-ingest', key
                              )
        if f.suffix.lower() == '.jp2':
            logger.info(f"Converting {f.name} from JP2 to JPG")
            try:
                from PIL import Image
                import io

                # Open and convert JP2 to JPG
                with Image.open(f) as img:
                    # Convert to RGB if necessary (JP2 might be in different color mode)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Create JPG filename
                    jpg_filename = f.stem + '.jpg'

                    # Save to bytes buffer
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=95)
                    buffer.seek(0)

                    # Upload JPG version
                    jpg_key = "/".join([accession_number,
                                        "SOURCE", "jpg", jpg_filename])
                    logger.info(f"Uploading JPG version: {jpg_filename}")

                    s3.upload_fileobj(buffer,
                                      'meadow-s-ingest', jpg_key
                                      )

            except Exception as e:
                logger.error(f"Failed to convert {f.name} to JPG: {e}")

    marker_ocr_tasks = []
    fws = []
    for i, spec in enumerate(specs):
        fw = Firework(
            tasks=PyTask(
                func="auxiliary.surya_on_image",
                inputs=["source_s3_key", "file_name", "accession_number"],
            ),
            spec=spec,
            name=f"Image {i:010d}"
        )

        fws.append(fw)

    ingest_sheet_fw = Firework(
        tasks=[PyTask(
            func="auxiliary.make_ingest_sheet",
            inputs=["file_names", "accession_number"],
        )],
        spec={
            "file_names": [f.get("file_name") for f in specs],
            "accession_number": accession_number
        }
    )
    fws.append(ingest_sheet_fw)
    wf = Workflow(
        fws,
        metadata={"accession_number": accession_number},
        name=accession_number,
        links_dict=None,
    )
    lp.add_wf(wf)
