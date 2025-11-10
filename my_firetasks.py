import shutil
from fireworks.core.firework import FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize


@explicit_serialize
class ImagesToPDF(FiretaskBase):
    """
    Wrapper around shutil.make_archive to make tar archives.

    Args:
        image_files (list[str]): Name of the file to create.
        format (str): Optional. one of "zip", "tar", "bztar" or "gztar".
    """

    _fw_name = "ImagesToPDF"
    required_params = ["image_files"]

    def run_task(self, fw_spec):
        print(self["image_files"])
