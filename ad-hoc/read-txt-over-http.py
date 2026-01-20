import requests

response = requests.get(
    "https://nu-impulse-production.s3.us-east-1.amazonaws.com/P0491_35556036056489/TXT/35556036056489_00000001.txt"
)
urls = [
    "https://nu-impulse-production.s3.us-east-1.amazonaws.com/P0491_35556036056489/TXT/35556036056489_00000001.txt",
    "https://nu-impulse-production.s3.us-east-1.amazonaws.com/P0491_35556036056489/TXT/35556036056489_00000002.txt",
    "https://nu-impulse-production.s3.us-east-1.amazonaws.com/P0491_35556036056489/TXT/35556036056489_00000003.txt",
    "https://nu-impulse-production.s3.us-east-1.amazonaws.com/P0491_35556036056489/TXT/35556036056489_00000004.txt",
]

docs = []

for url in urls:
    docs.append(requests.get(url).content)

print(docs)
