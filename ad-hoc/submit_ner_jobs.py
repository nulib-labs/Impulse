import boto3
import os

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY")
)

bucket = s3(bucket = "nu-impulse-production")
