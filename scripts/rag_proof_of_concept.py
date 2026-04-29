import argparse
import os
import sys
import textwrap

import certifi
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INDEX = "developmentVectorSearchIndex"
EMBEDDING_PATH = "embedding"
DATABASE = "praxis"
COLLECTION = "embeddings"


def encode_query(query: str) -> list[float]:
    """Encode a query string into an embedding vector using the Qwen model on CPU."""
    print(f"Loading model {MODEL_NAME} (CPU)...")
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": "float32"},
    )
    print("Encoding query...")
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()


def build_pipeline(
    query_vector: list[float],
    index_name: str,
    limit: int,
    num_candidates: int,
    impulse_identifier: str | None = None,
) -> list[dict]:
    """Build the MongoDB $vectorSearch aggregation pipeline."""
    vector_search_stage: dict = {
        "$vectorSearch": {
            "index": index_name,
            "path": EMBEDDING_PATH,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }

    # Optional pre-filter on impulse_identifier
    if impulse_identifier:
        vector_search_stage["$vectorSearch"]["filter"] = {
            "impulse_identifier": impulse_identifier,
        }

    # Project the fields we care about plus the search score
    project_stage = {
        "$project": {
            "_id": 0,
            "chunk": 1,
            "impulse_identifier": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    }

    return [vector_search_stage, project_stage]


def run_search(
    uri: str,
    pipeline: list[dict],
) -> list[dict]:
    """Execute the vector search pipeline and return results."""
    client = MongoClient(uri, tlsCAFile=certifi.where())
    db = client[DATABASE]
    coll = db[COLLECTION]

    results = list(coll.aggregate(pipeline))
    client.close()
    return results


def print_results(results: list[dict], query: str) -> None:
    """Pretty-print search results to the terminal."""
    print(f"\n{'=' * 72}")
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'=' * 72}\n")

    if not results:
        print("No results found.")
        return

    for i, doc in enumerate(results, start=1):
        score = doc.get("score", "N/A")
        identifier = doc.get("impulse_identifier", "unknown")
        chunk = doc.get("chunk", "")

        # Wrap long chunks for readability
        wrapped = textwrap.fill(
            chunk, width=80, initial_indent="  ", subsequent_indent="  "
        )

        print(f"[{i}] score={score:.4f}  identifier={identifier}")
        print(wrapped)
        print()


def get_mongo_uri(production: bool) -> str:
    """Resolve the MongoDB connection URI from environment variables."""
    if production:
        uri = os.getenv("IMPULSE_MONGODB_URI")
        label = "production"
    else:
        uri = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
        label = "debug/staging"

    if not uri:
        env_var = "IMPULSE_MONGODB_URI" if production else "IMPULSE_MONGODB_URI_DEBUG"
        print(
            f"Error: {env_var} is not set. Cannot connect to the {label} database.",
            file=sys.stderr,
        )
        sys.exit(1)

    return uri


def main(argv: list[str] | None = None) -> None:
    from ollama import chat

    env = "debug/staging"
    print(f"Target: {env}")

    uri = get_mongo_uri(production=False)

    messages = []
    while True:
        message = input("Query: ")
        message_vector = encode_query(message)
        pipeline = build_pipeline(
            query_vector=message_vector,
            index_name=DEFAULT_INDEX,
            limit=5,
            num_candidates=10,
        )
        messages.append({"role": "user", "content": message, "stream": True})
        messages.append(
            {
                "role": "system",
                "content": f"These are the returned most relevant documents from a vector search in MongoDB. You should know that all documents in this database are Environmental Impact Statements housed at Northwestern University.\n{run_search(uri, pipeline)}",
                "stream": True,
            }
        )
        stream = chat(model="llama3.2:1b", messages=messages, stream=True)
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)


main()
