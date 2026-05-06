from fireworks import Firework, LaunchPad
from tasks.my_tasks import DocumentExtractionTask
from tasks.config import MONGO_URI
import boto3
import subprocess
import os
import certifi

DEBUG = False

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

out = subprocess.Popen(
    ["aws", "s3", "ls", "--profile", "impulse", "s3://nu-impulse-production"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
stdout, stderr = out.communicate()

if stderr:
    print("Error:", stderr.decode("utf-8"))
else:
    output = stdout.decode("utf-8").splitlines()
    output = [i.strip().replace("PRE ", "") for i in output if i.strip()]
z = 0
for i in output:
    if not i.startswith("p1190"):
        continue

    launchpad: LaunchPad = LaunchPad(
        uri_mode=True,
        host=MONGO_URI,
        name="fireworks",
        mongoclient_kwargs={"tlsCAFile": certifi.where()},
    )

    session = boto3.Session(profile_name="impulse")
    client = session.client("s3", region_name="us-west-2")
    paginator = client.get_paginator("list_objects_v2")
    prefix = f"{i}"
    operation_parameters = {
        "Bucket": "nu-impulse-production",
        "Prefix": prefix,
    }

    page_iterator = paginator.paginate(**operation_parameters)
    impulse_keys: list[str] = []
    for page in page_iterator:
        try:
            for j in page["Contents"]:
                key = f"s3://nu-impulse-production/{j['Key']}"
                if key.endswith("jpg") and "unprocessed" in key:
                    impulse_keys.append(key)
                    print(key)
        except:
            continue

    ocr_fw: Firework = Firework(
        DocumentExtractionTask(),
        spec={
            "impulse_identifier": i.replace("/", "").lower(),
            "find_path_array_in": "keys",
            "keys": impulse_keys,
        },
        name="Document Extraction Workflow",
    )

    launchpad.add_wf(ocr_fw)
