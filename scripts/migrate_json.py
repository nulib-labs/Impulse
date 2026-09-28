#!/usr/bin/env python3
import re, sys
import boto3
from concurrent.futures import ThreadPoolExecutor

BUCKET = "impulse-data-prod"
PROFILE = "mellon-account"
DRY_RUN = "--run" not in sys.argv   # pass --run to actually move

s3 = boto3.Session(profile_name=PROFILE).client("s3")
pat = re.compile(r"^jobs/[^/]+/[^/]+/[^/]+\.json$")

keys = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="jobs/"):
    keys += [o["Key"] for o in page.get("Contents", []) if pat.match(o["Key"])]

print(f"{len(keys)} files to move")

def dest(key):
    d, f = key.rsplit("/", 1)
    return f"{d}/json/{f}"

def copy(key):
    if DRY_RUN:
        print(f"(dryrun) {key} -> {dest(key)}")
        return
    s3.copy_object(Bucket=BUCKET, Key=dest(key),
                   CopySource={"Bucket": BUCKET, "Key": key})

with ThreadPoolExecutor(max_workers=32) as ex:
    list(ex.map(copy, keys))

if not DRY_RUN:
    for i in range(0, len(keys), 1000):   # delete_objects takes up to 1000
        batch = [{"Key": k} for k in keys[i:i+1000]]
        s3.delete_objects(Bucket=BUCKET, Delete={"Objects": batch})
    print("done")
