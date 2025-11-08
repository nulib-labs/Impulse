import os
import pathlib
from loguru import logger
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.filepad import FilePad


# Return a list of all image files in a directory
def list_image_files_in_dir(x: pathlib.Path) -> list[pathlib.Path]:
    image_suffixes = (".jpg", ".png", ".jp2", ".jpeg")

    # Ensure x is a directory
    if not x.is_dir():
        logger.error(f"The path you provided:\n{x} is not a directory.")
        return []

    # Return absolute paths to image files
    return [
        (x / f).resolve() for f in os.listdir(x) if f.lower().endswith(image_suffixes)
    ]


def get_fireworks_data():
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
    return {"LaunchPad": lp, "FilePad": fp}
