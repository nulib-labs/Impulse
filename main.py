from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask
import glob
import argparse
from tqdm import tqdm
from fabric import Connection

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
    logdir="./logs",
)
fp = FilePad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    logdir="./logs",
)

if do_reset:
    lp.reset("2025-10-08", require_password=False)
    fp.reset()

suffixes = ["jpg", "png", "jpeg", "tiff", "jp2", "xml"]

files = []

if is_recursive:
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(input_path, "**", f"*.{suffix}"), recursive=True)
        )
    files = sorted(files)
    print(files)
    pass

else:
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(input_path, "**", f"*.{suffix}"), recursive=True)
        )
    print(files)
    identifiers = []
    xml_identifier = None
    for f in tqdm(sorted(files), desc="Uploading files..."):
        name = f.split("/")[-1]
        new_identifier = str(accession_number) + "_" + str(name).zfill(10)
        if not f.endswith(".xml"):
            file_id, _ = fp.add_file(
                f,
                identifier=new_identifier,
                metadata={
                    "source_path": f,
                    "filename": name,
                    "accession_number": accession_number,
                },
            )
            identifiers.append(new_identifier)
        else:
            print("Found metadata xml file:")
            file_id, _ = fp.add_file(
                f,
                identifier=new_identifier,
                metadata={
                    "source_path": f,
                    "filename": name,
                    "accession_number": accession_number,
                },
            )
            xml_identifier = file_id
    if xml_identifier is not None:
        spec = {
            "identifiers": identifiers,
            "accession_number": accession_number,
            "xml_identifier": xml_identifier,
        }

    else:
        spec = {"identifiers": identifiers, "accession_number": accession_number}

    if xml_identifier is not None:
        marker_ocr_tasks = []
        for identifier in spec["identifiers"]:
            marker_ocr_tasks.append(
                PyTask(
                    func="auxiliary.marker_on_image",
                    inputs=["identifier", "accession_number"],
                    outputs="converted_images",
                )
            )
        fw1 = Firework(tasks=marker_ocr_tasks)
        wf = Workflow(fireworks=[fw1])
        wf = Workflow(
            [fw1],
            metadata={"accession_number": accession_number},
            name=accession_number,
            links_dict=None,
        )
    else:
        marker_ocr_tasks = []
        for identifier in identifiers:
            marker_ocr_tasks.append(
                PyTask(
                    func="auxiliary.marker_on_image",
                    inputs=["identifier", "accession_number"],
                    outputs="page_schema",
                    spec={
                        "identifier": identifier,
                        "accession_number": accession_number,
                    },
                )
            )
        fw1 = Firework(
            tasks=marker_ocr_tasks,
            name=accession_number,
            spec={"identifier": identifier, "accession_number": accession_number},
        )

        wf = Workflow(fireworks=[fw1], name="My test")
        lp.add_wf(wf)
