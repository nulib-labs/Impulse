import certifi
import os
from pymongo import MongoClient
from fireworks import Firework, LaunchPad, Workflow
from tasks.my_tasks import EmbeddingTask

DEBUG = True

MONGO_URI = (
    os.getenv("IMPULSE_MONGODB_URI_DEBUG")
    if DEBUG
    else os.getenv("IMPULSE_MONGODB_URI")
)

# -----------------------------
# Mongo connection
# -----------------------------
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["praxis"]

# -----------------------------
# Get all unique identifiers
# -----------------------------
impulse_ids = db["colt"].distinct("impulse_identifier")

impulse_ids = [i for i in impulse_ids if isinstance(i, str) and i.strip()]
impulse_ids = [impulse_ids[0]]
print(f"Found {len(impulse_ids)} identifiers")

# -----------------------------
# Fireworks LaunchPad
# -----------------------------
launchpad = LaunchPad(
    uri_mode=True,
    host=MONGO_URI,
    name="fireworks",
    mongoclient_kwargs={"tlsCAFile": certifi.where()},
)

# -----------------------------
# Create Fireworks
# -----------------------------
fireworks = [
    Firework(
        EmbeddingTask(),
        spec={
            "impulse_identifier": impulse_id,
        },
        name=f"Embedding: {impulse_id}",
    )
    for impulse_id in impulse_ids
]

# -----------------------------
# Submit workflow
# -----------------------------
for f in fireworks:
    launchpad.add_wf(f)

print("Workflow submitted.")
