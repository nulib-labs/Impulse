import os

import certifi
from pymongo import MongoClient

DEBUG = False  # Defines which environment variables to use. Should not be used in prod

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["praxis"]
collection = db["colt"]

# -------------------------------------------------------------------------
# The ocr_confidences field is stored as nested objects (not arrays) because
# tasks.helpers.funcs.stringify_keys() converts every list into an object
# with stringified numeric keys, e.g.:
#
#   ocr_confidences: {
#     "0": {                            <-- page (was list index)
#       "0": {                          <-- line (was list index)
#         "text": "...",
#         "confidence": 0.95,
#         "chars": {                    <-- chars (was list)
#           "0": {"text": "a", "confidence": 0.9},
#           "1": {"text": "b", "confidence": 0.8}
#         }
#       }, ...
#     }, ...
#   }
#
# We use $objectToArray at each nesting level to convert these back into
# arrays that $unwind can operate on.
# -------------------------------------------------------------------------

pipeline = [
    # 1. Only process documents that actually have confidence data
    {"$match": {"ocr_confidences": {"$exists": True, "$ne": {}}}},
    # 2. Convert page-level object -> array of {k: page_id, v: lines_obj}
    {
        "$addFields": {
            "_pages": {"$objectToArray": "$ocr_confidences"},
        }
    },
    # 3. Unwind pages
    {"$unwind": "$_pages"},
    # 4. Convert line-level object -> array of {k: line_idx, v: line_data}
    #    and compute per-line avg char confidence *inline* using $map + $avg
    #    to avoid an expensive char-level $unwind (~108M intermediate docs).
    {
        "$addFields": {
            "_lines": {
                "$map": {
                    "input": {"$objectToArray": "$_pages.v"},
                    "as": "line",
                    "in": {
                        "lineConfidence": "$$line.v.confidence",
                        "avgCharConfidence": {
                            "$avg": {
                                "$map": {
                                    "input": {"$objectToArray": "$$line.v.chars"},
                                    "as": "ch",
                                    "in": "$$ch.v.confidence",
                                }
                            }
                        },
                    },
                }
            },
        }
    },
    # 5. Per-document: compute line count and averages from the _lines array
    {
        "$group": {
            "_id": "$_id",
            "lineCount": {"$sum": {"$size": "$_lines"}},
            "avgLineConfidence": {
                "$avg": {"$avg": "$_lines.lineConfidence"},
            },
            "avgCharConfidence": {
                "$avg": {"$avg": "$_lines.avgCharConfidence"},
            },
        }
    },
    # 6. Final aggregation across all documents
    {
        "$group": {
            "_id": None,
            "docCount": {"$sum": 1},
            "avgLinesPerDoc": {"$avg": "$lineCount"},
            "avgLineConfidence": {"$avg": "$avgLineConfidence"},
            "avgCharConfidence": {"$avg": "$avgCharConfidence"},
        }
    },
]

result = list(collection.aggregate(pipeline, allowDiskUse=True))

if result:
    r = result[0]
    print(f"Documents processed:  {r['docCount']}")
    print(f"Avg Lines per Doc:    {r['avgLinesPerDoc']:.4f}")
    print(f"Avg Line Confidence:  {r['avgLineConfidence']:.4f}")
    print(f"Avg Char Confidence:  {r['avgCharConfidence']:.4f}")
else:
    print("No results found.")
