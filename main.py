from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask
import glob


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

barcode_dir = "./pilot_testing/p1074_35556032756942/"

file_path = os.path.join(barcode_dir, "JPG_OG")
barcode = os.path.basename(os.path.normpath(barcode_dir))
files = sorted(glob.glob(os.path.join(file_path, "*.jpg")))
identifiers = []
for f in sorted(files):
    print(f)
    name = f.split("/")[-1]
    barcode_name = barcode_dir.split("/")[-1]
    file_id, identifier = fp.add_file(
        f,
        identifier=str(barcode_name + name),
        metadata={"source_path": f, "barcode": barcode, "filename": name},
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
