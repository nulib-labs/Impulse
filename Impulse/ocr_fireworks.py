from pathlib import PosixPath
from fireworks.core.firework import Firework
from fireworks.core.launchpad import LaunchPad
from fireworks.user_objects.firetasks.script_task import PyTask
import os

from fireworks.utilities.filepad import FilePad
from tqdm import tqdm


def define_image_processing_spec(accession_number, files):
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

    suffixes = ["jpg", "png", "jpeg", "tiff", "jp2", "xml"]
    print(files)
    identifiers = []
    for f in tqdm(sorted(files), desc="Uploading files..."):
        f: PosixPath
        absolute_path = str(f)
        filename = str(f.name)
        new_identifier = str(accession_number) + "_" + str(filename).zfill(10)
        file_id, _ = fp.add_file(
            f,
            identifier=new_identifier,
            metadata={
                "source_path": absolute_path,
                "filename": filename,
                "accession_number": accession_number,
            },
        )
        identifiers.append(new_identifier)
    spec = {
        "identifiers": identifiers,
        "accession_number": accession_number,
    }

    return spec


def define_image_processing_firework(name: str, spec: dict, files: list):
    fw = Firework(
        [
            PyTask(
                func="fireworks_auxiliary.image_conversion_task",
                inputs=[
                    "identifiers",
                    "accession_number",
                ],
                outputs="converted_images",
            ),
        ],
        name="",
        spec=define_image_processing_spec("XXXX", files),
    )

    return fw
