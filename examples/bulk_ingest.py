import boto3


def list_pdfs(bucket_name, prefix=""):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    pdf_paths = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".pdf"):
                pdf_paths.append(f"s3://{bucket_name}/{key}")

    return pdf_paths


import yaml

data = {
    "name": "Warlord",
    "fws": [],
    "links": [],
    "metadata": {},
}
paths = list_pdfs("nu-impulse-production", prefix="")
for i, path in enumerate(paths, start=1):
    accession_number = path.split("/")[-2]

    fw = {
        "fw_id": i,
        "name": "Extraction",
        "spec": {
            "accession_number": accession_number,
            "find_path_array_in": "path_array",
            "_tasks": [{"_fw_name": "Document Extraction Task"}],
            "path_array": [path],
        },
    }

    data["fws"].append(fw)

with open("my_file.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)
