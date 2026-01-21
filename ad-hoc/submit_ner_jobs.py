import boto3
from fireworks.core.launchpad import LaunchPad
import os
from fireworks.user_objects.firetasks.script_task import PyTask
from fireworks.core.firework import Firework, Workflow
import botocore

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
)


conn_str = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
lp = LaunchPad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    name="fireworks",
)

bucket = "nu-impulse-production"
paginator = s3.get_paginator("list_objects_v2")

# 1. Get top-level directories
dirs: list[str] = []

for page in paginator.paginate(Bucket=bucket, Prefix="", Delimiter="/"):
    for d in page.get("CommonPrefixes", []):
        dirs.append(d["Prefix"])

# 2. For each directory, list .txt files in TXT/
txt_files = []

for d in dirs[0]:
    txt_prefix = f"{d}TXT/"
    fws = []
    accession_number = d.replace("/", "")
    for page in paginator.paginate(Bucket=bucket, Prefix=txt_prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            if key.endswith(".txt"):
                s3_key = key
                filename: str = key.split("/")[-1]
                fw = Firework(
                    tasks=PyTask(
                        func="auxiliary.spacy_experiment",
                        inputs=["s3_key", "filename", "accession_number"],
                    ),
                    spec={
                        "s3_key": key,
                        "filename": key,
                        "accession_number": accession_number,
                    },
                    name=f"SPACY_NER {accession_number}",
                )

                fws.append(fw)
    print(f"submitted job with {len(fws)} fireworks.")

    wf = Workflow(fireworks=fws, name=f"Ad-Hoc NER {accession_number}")
    _ = lp.add_wf(wf)
