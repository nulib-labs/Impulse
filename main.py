from os.path import isdir
from fireworks.core.fworker import FWorker
from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import QUEUEADAPTER_LOC, RAPIDFIRE_SLEEP_SECS, os
from fireworks.utilities.filepad import FilePad
import certifi
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask, ScriptTask
import glob
import uuid
import auxiliary
from fireworks.queue.queue_launcher import rapidfire
from fireworks.queue.queue_adapter import QueueAdapterBase

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

fp.reset()
lp.reset(password="2025-10-03")
barcode_dir = "./data/raw/"
for d in os.listdir(barcode_dir):
    if d == ".DS_Store":
        continue
    file_path = os.path.join(barcode_dir, d)
    file_path = os.path.join(file_path, "JP2000")

    files = sorted(glob.glob(os.path.join(file_path, "*.jp2")))
    files = files[:10]
    identifiers = []
    for f in sorted(files):
        file_id, identifier = fp.add_file(f, identifier=str(uuid.uuid4()))
        print(f"{identifier}")
        identifiers.append(identifier)

    fw = Firework(
        [
            PyTask(
                func="auxiliary.image_conversion_task",
                inputs=[
                    "identifiers",
                    "barcode_dir",
                ],  # Looking for key 'identifiers'
                outputs="converted_images",
            ),
            PyTask(
                func="auxiliary.image_to_pdf",
                inputs=["converted_images", "barcode_dir"],
                outputs="PDF_id",
            ),
            PyTask(func="auxiliary.marker_on_pdf", inputs=["PDF_id"]),
        ],
        spec={"identifiers": identifiers, "barcode_dir": barcode_dir},
        name="OCR Firework",
    )
    wf = Workflow([fw])

    lp.add_wf(wf)
    break
