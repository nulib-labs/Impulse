from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask
import glob
import argparse

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
)

parser.add_argument(
    "--identifier",
    "-id",
    help="The identifier for which the files should be linked. This should be a barcode.",
    action="store_true",
)

parser.add_argument(
    "--fw_name",
    "-n",
    help="A custom name of the workflow. If no argument is passed, defaults to value of --identifier",
    required=False,
)

args = parser.parse_args()

input_path = args.input
is_recursive = args.recursive
identifier = args.identifier


conn_str: str
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
    for f in sorted(files):
        print(f)
        name = f.split("/")[-1]
        file_id, identifier = fp.add_file(
            f,
            identifier=str(identifier + name),
            metadata={"source_path": f, "identifier": identifier, "filename": name},
        )
        print(f"{identifier}")
        identifiers.append(identifier)

    fw = Firework(
        [
            PyTask(
                func="auxiliary.image_conversion_task",
                inputs=[
                    "identifiers",
                    "barcode_dir",
                ],
                outputs="converted_images",
            ),
            PyTask(
                func="auxiliary.image_to_pdf",
                inputs=["converted_images", "barcode_dir"],
                outputs="PDF_id",
            ),
            PyTask(func="auxiliary.marker_on_pdf", inputs=["PDF_id"]),
        ],
        name=identifier,
        spec={"identifiers": identifiers, "barcode_dir": barcode_dir},
    )
    wf = Workflow([fw], metadata={"barcode": barcode}, name=(barcode + "_workflow"))

    lp.add_wf(wf)
