import boto3

session = boto3.Session(profile_name="impulse")

s3: boto3.client.S3 = session.client("s3")

s3.list_objects_v

print(type(s3))
print(s3)
