import boto3

session = boto3.Session(profile_name="impulse")
s3_client = session.client("s3")
paginator = s3_client.get_paginator("list_objects_v2")
pages = paginator.paginate(
    Bucket="nu-impulse-production", Prefix="p0491p1074eis-1766005955"
)

bucket = "nu-impulse-production"

for page in pages:
    if "Contents" not in page:
        continue
    for obj in page["Contents"]:
        key = obj["Key"]
        if key.upper().endswith(".TXT"):
            # Extract the accession number (second part of the path)
            parts = key.split("/")
            if len(parts) >= 2:
                accession_number = parts[1]
                # Create new key by removing the prefix directory
                new_key = "/".join(parts[1:])

                print(f"Moving: {key} -> {new_key}")

                # Copy object to new location
                s3_client.copy_object(
                    Bucket=bucket,
                    CopySource={"Bucket": bucket, "Key": key},
                    Key=new_key,
                )

                # Delete original object
                s3_client.delete_object(Bucket=bucket, Key=key)

                print(f"Moved successfully")
