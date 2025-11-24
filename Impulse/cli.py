from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad


def define_pads():
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

    return lp, fp
