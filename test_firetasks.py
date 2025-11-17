from fireworks.core.launchpad import Firework, LaunchPad, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask, ScriptTask
import os

conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
fw1 = Firework(
    [
        PyTask(
            func="my_firetasks.ImagesToPDF",
            kwargs={"image_files": ["./pilot_test_data/p1074_35556032756942/5.jpg"]},
        )
    ],
    name="convert_images_to_pdf",
)

wf = Workflow([fw1])
lp = LaunchPad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    name="fireworks",
)
lp.add_wf(wf)
