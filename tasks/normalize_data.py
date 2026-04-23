from pymongo import MongoClient, UpdateOne, DeleteOne
import os
import certifi
from tqdm import tqdm

client = MongoClient(os.getenv("IMPULSE_MONGODB_URI"), tlsCAFile=certifi.where())

db = client["praxis"]
coll = db["colt"]

# -----------------------------
# 1. Lowercase impulse_identifier
# -----------------------------
updates = []

cursor = coll.find(
    {"impulse_identifier": {"$type": "string"}}, {"impulse_identifier": 1}
)

for doc in tqdm(cursor):
    original = doc["impulse_identifier"]
    lowered = original.lower()

    if lowered != original:
        updates.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": {"impulse_identifier": lowered}})
        )

    if len(updates) >= 1000:
        coll.bulk_write(updates, ordered=False)
        updates = []

if updates:
    coll.bulk_write(updates, ordered=False)

print("Lowercasing complete.")

# -----------------------------
# 2. Remove duplicates
# -----------------------------

pipeline = [
    {
        "$group": {
            "_id": {
                "impulse_identifier": "$impulse_identifier",
                "page_number": "$page_number",
            },
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1},
        }
    },
    {"$match": {"count": {"$gt": 1}}},
]

duplicates = coll.aggregate(pipeline, allowDiskUse=True)

deletes = []

for group in duplicates:
    ids = group["ids"]

    # keep the first document, delete the rest
    for _id in ids[1:]:
        deletes.append(DeleteOne({"_id": _id}))

    if len(deletes) >= 1000:
        coll.bulk_write(deletes, ordered=False)
        deletes = []

if deletes:
    coll.bulk_write(deletes, ordered=False)

print("Deduplication complete.")
