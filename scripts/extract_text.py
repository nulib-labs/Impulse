import boto3
import os
import json
from bs4 import BeautifulSoup
from pathlib import Path

from tqdm import tqdm

S3_BUCKET = os.environ["S3_BUCKET"]
AWS_PROFILE = os.environ["AWS_PROFILE"]
AWS_REGION = os.environ["AWS_REGION"]
        
session = boto3.Session(
    profile_name=AWS_PROFILE,
    region_name=AWS_REGION,
)
s3 = session.client("s3")

def handle_txt_format(key):
    try:

        response = s3.get_object(Bucket=S3_BUCKET, Key=key)

# Read the body content and load it as JSON
        file_content = response['Body'].read().decode('utf-8')
        json_data = json.loads(file_content)
        ocr_data: dict = json_data.get("ocr_data")
        payload = []
        for block in ocr_data["blocks"]:
            html_raw = block.get("html", "")
            soup = BeautifulSoup(html_raw, "html.parser")
            soup2 = soup.get_text()
            payload.append(soup2)
        

        payload = "\n".join(payload)
    
        out_key =key.replace("json", "txt")
        s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )
    except Exception as e:
        raise e

def generate_json_files():
    paginator = s3.get_paginator('list_objects_v2') #
    pages = paginator.paginate(Bucket="impulse-data-prod", Prefix="jobs/p1274/")
    extension = "json"
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Filter by extension client-side
                if key.lower().endswith(extension.lower()):
                    yield key 

def main():
    for key in tqdm(generate_json_files()):
        handle_txt_format(key)

main()
