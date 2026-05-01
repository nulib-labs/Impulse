from fireworks import Firework, LaunchPad
from tasks.my_tasks import ImageProcessingTask
from tasks.config import MONGO_URI
import boto3
import subprocess
import os
import certifi

DEBUG = True

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")


launchpad: LaunchPad = LaunchPad(
    uri_mode=True,
    host=MONGO_URI,
    name="fireworks",
    mongoclient_kwargs={"tlsCAFile": certifi.where()},
)


ocr_fw: Firework = Firework(
    ImageProcessingTask(),
    spec={
        "impulse_identifier": "TEST_APRIL",
        "find_path_array_in": "keys",
        "keys": ["s3://nu-impulse-production/TEST_APRIL/limb-00000005.jpg"],
    },
    name="Image Processing Test April",
)

launchpad.add_wf(ocr_fw)
