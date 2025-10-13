from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask
import glob
import argparse
from tqdm import tqdm


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
    "--identifier",
    "-id",
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
is_recursive = args.recursive
identifier = args.identifier
fw_name = args.fw_name
do_reset = args.reset


conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
lp = LaunchPad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    name="fireworks",
    logdir="./logs",
)
fp = FilePad(
    host=(conn_str + "/fireworks?"),
    port=27017,
    uri_mode=True,
    database="fireworks",
)

if do_reset:
    lp.reset("2025-10-08", require_password=False)
    fp.reset()

suffixes = ["jpg", "png", "jpeg", "tiff", "jp2"]

files = []

if is_recursive:
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(input_path, "**", f"*.{suffix}"), recursive=True)
        )
    files = sorted(files)
    pass

else:
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(input_path, "**", f"*.{suffix}"), recursive=True)
        )
    identifiers = []
    for f in tqdm(sorted(files), desc="Uploading files..."):
        name = f.split("/")[-1]
        new_identifier = str(identifier) + name
        file_id, _ = fp.add_file(
            f,
            identifier=new_identifier,
            metadata={"source_path": f, "identifier": identifier, "filename": name},
        )
        identifiers.append(new_identifier)
    fw = Firework(
        [
            PyTask(
                func="auxiliary.image_conversion_task",
                inputs=[
                    "identifiers",
                    "identifier",
                ],
                outputs="converted_images",
            ),
            PyTask(
                func="auxiliary.image_to_pdf",
                inputs=["converted_images", "identifier"],
                outputs="PDF_id",
            ),
            PyTask(func="auxiliary.marker_on_pdf", inputs=["PDF_id", "identifier"]),
        ],
        name=fw_name,
        spec={"identifiers": identifiers, "identifier": identifier},
    )
    wf = Workflow([fw], metadata={"identifier": identifier}, name=identifier)

    lp.add_wf(wf)
