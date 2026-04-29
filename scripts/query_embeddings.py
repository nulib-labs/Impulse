"""Semantic search over the Impulse embeddings collection.

Encodes a natural-language query with the same Qwen/Qwen3-Embedding-0.6B model
used by the EmbeddingTask, then runs a $vectorSearch aggregation against the
praxis.embeddings collection on MongoDB Atlas.

Usage examples:
    python scripts/query_embeddings.py "Where is JFK Airport?"
    python scripts/query_embeddings.py "Civil War battles" --limit 5
    python scripts/query_embeddings.py "labor statistics" --prod
    python scripts/query_embeddings.py "shipping routes" --identifier p0491_35556036056489

Requires:
    - IMPULSE_MONGODB_URI_DEBUG (staging) or IMPULSE_MONGODB_URI (production)
      environment variables to be set.
    - A MongoDB Atlas Vector Search index (default: developmentVectorSearchIndex)
      configured on praxis.embeddings with path "embedding".
"""

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic search over Impulse embeddings (MongoDB Atlas Vector Search).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s "Where is JFK Airport?"
              %(prog)s "Civil War battles" --limit 5
              %(prog)s "labor statistics" --prod
              %(prog)s "shipping routes" --identifier p0491_35556036056489
        """),
    )
    parser.add_argument(
        "query",
        help="Natural-language query text to search for.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of results to return (default: 10).",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="numCandidates for vector search — higher values improve recall "
        "at the cost of latency (default: 100).",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        default=False,
        help="Query the production database instead of the debug/staging database.",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default=None,
        metavar="ID",
        help="Filter results to a specific impulse_identifier.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=DEFAULT_INDEX,
        help=f"Atlas Vector Search index name (default: {DEFAULT_INDEX}).",
    )
    return parser.parse_args(argv)


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
        print(f"Error: {env_var} is not set. Cannot connect to the {label} database.", file=sys.stderr)
        sys.exit(1)

    return uri


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
        wrapped = textwrap.fill(chunk, width=80, initial_indent="  ", subsequent_indent="  ")

        print(f"[{i}] score={score:.4f}  identifier={identifier}")
        print(wrapped)
        print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    env = "production" if args.prod else "debug/staging"
    print(f"Target: {env}")

    # 1. Encode the query
    query_vector = encode_query(args.query)
    print(f"Embedding dimension: {len(query_vector)}")

    # 2. Connect and search
    uri = get_mongo_uri(args.prod)
    pipeline = build_pipeline(
        query_vector=query_vector,
        index_name=args.index,
        limit=args.limit,
        num_candidates=args.candidates,
        impulse_identifier=args.identifier,
    )

    print(f"Searching (index={args.index}, limit={args.limit}, candidates={args.candidates})...")
    results = run_search(uri, pipeline)

    # 3. Display
    print_results(results, args.query)


if __name__ == "__main__":
    main()
