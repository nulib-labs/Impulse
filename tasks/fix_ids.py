import certifi
from pymongo import MongoClient, UpdateOne
import os
from tqdm import tqdm

source_client = MongoClient(
    os.getenv("IMPULSE_MONGODB_URI_DEBUG"), tlsCAFile=certifi.where()
)
target_client = MongoClient(os.getenv("IMPULSE_MONGODB_URI"), tlsCAFile=certifi.where())

source = source_client["praxis"]["colt"]
target = target_client["praxis"]["colt"]

operations = []

for doc in tqdm(source.find()):
    operations.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

    if len(operations) >= 1000:
        target.bulk_write(operations)
        operations = []

if operations:
    target.bulk_write(operations)
