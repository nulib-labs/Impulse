from sentence_transformers import SentenceTransformer
from pymongo.mongo_client import MongoClient
import certifi

# --- Clients ---
client = MongoClient(
    "mongodb+srv://aerithnetzer:aK6DdUXcjxkqfgj2@impulse-staging-1.pfjqa.mongodb.net/?appName=Impulse-Staging-1",
    tlsCAFile=certifi.where(),
)
db = client["praxis"]
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


def replace_keys():
    db.colt.update_many(
        {},
        [
            {
                "$set": {
                    "impulse_identifier": {
                        "$replaceAll": {
                            "input": {
                                "$replaceAll": {
                                    "input": {
                                        "$replaceAll": {
                                            "input": {
                                                "$replaceAll": {
                                                    "input": {
                                                        "$toString": "$impulse_identifier"
                                                    },
                                                    "find": "{",
                                                    "replacement": "",
                                                }
                                            },
                                            "find": "}",
                                            "replacement": "",
                                        }
                                    },
                                    "find": "'",
                                    "replacement": "",
                                }
                            },
                            "find": '"',
                            "replacement": "",
                        }
                    }
                }
            }
        ],
    )

    client.close()


def embed_query(text: str) -> list[float]:
    """Embed a query string using Qwen3-Embedding-0.6B."""
    return model.encode(text, prompt_name="query").tolist()


def vector_search(query_text: str, top_k: int = 10) -> list[dict]:
    """
    Embed query_text and run a vector search against the MongoDB vector index.
    Returns results with similarity scores.
    """
    query_vector = embed_query(query_text)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "developmentVectorSearchIndex",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 1,
                "score": {"$meta": "vectorSearchScore"},
                "embedding": 0,  # exclude the raw vector from results
                # Uncomment fields you want returned:
                # "title": 1,
                # "text": 1,
                # "metadata": 1,
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def print_results(results: list[dict]) -> None:
    """Pretty-print results with scores."""
    if not results:
        print("No results found.")
        return
    for i, doc in enumerate(results, 1):
        score = doc.pop("score", None)
        print(f"[{i}] score={score:.4f}  {doc}")


# --- Run it ---
if __name__ == "__main__":
    replace_keys()
    exit()
    query = "Where is the JFK space center?"  # <-- change this
    results = vector_search(query, top_k=5)
    print_results(results)
    client.close()
