from fireworks import Firework, LaunchPad, Workflow
from fireworks.core.rocket_launcher import launch_rocket
from tasks.mets import METSXMLToHathiTrustManifestTask
from tasks.my_tasks import DocumentExtractionTask
from tasks.config import MONGO_URI
import boto3
import subprocess
import os

DEBUG = True

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

out = subprocess.Popen(
    ["aws", "s3", "ls", "--profile", "myprofile", "s3://nu-impulse-production"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
stdout, stderr = out.communicate()

if stderr:
    print("Error:", stderr.decode("utf-8"))
else:
    output = stdout.decode("utf-8").splitlines()
    output = [i.strip().replace("PRE ", "") for i in output if i.strip()]
print(MONGO_URI)
for i in output:
    if "DATA" in i or i.startswith("J"):
        continue

    launchpad: LaunchPad = LaunchPad(
        uri_mode=True,
        host=MONGO_URI,
        name="fireworks",
        mongoclient_kwargs={"tlsCAFile": "/etc/ssl/certs/ca-bundle.crt"},
    )

    session = boto3.Session(profile_name="myprofile")
    client = session.client("s3", region_name="us-west-2")
    paginator = client.get_paginator("list_objects_v2")
    operation_parameters = {
        "Bucket": "nu-impulse-production",
        "Prefix": f"{i}SOURCE/jpg/",
    }

    page_iterator = paginator.paginate(**operation_parameters)
    impulse_keys: list[str] = []
    xml_key = f"s3://nu-impulse-production/{i}mets.xml"
    for page in page_iterator:
        try:
            for j in page["Contents"]:
                print(j["Key"])
                impulse_keys.append(f"s3://nu-impulse-production/{j['Key']}")
        except:
            continue

    print(i.replace("/", ""))
    ocr_fw: Firework = Firework(
        DocumentExtractionTask(),
        spec={
            "impulse_identifier": {i.replace("/", "")},
            "find_path_array_in": "keys",
            "keys": sorted(impulse_keys),
        },
        name="Document Extraction Workflow",
    )

    mets_fw: Firework = Firework(
        METSXMLToHathiTrustManifestTask(),
        spec={
            "impulse_identifier": {i.replace("/", "")},
            "s3_xml_path": xml_key,
            "s3_yaml_path": xml_key.replace(".xml", ".yaml"),
        },
    )

    wf: Workflow = Workflow([ocr_fw, mets_fw])

    launchpad.add_wf(ocr_fw)

    break
