import json
from pathlib import Path
from typing import List, Iterable
import os
import boto3
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
BUCKET_NAME = "nu-impulse-production"
PREFIX = ""
BATCH_SIZE = 32  # GPU batch size
KEY_CHUNK_SIZE = 512  # <-- what you asked for
AWS_PROFILE = "impulse"
MODEL_NAME = "nvidia/llama-embed-nemotron-8b"

session = boto3.Session(
    aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
)
s3 = session.client("s3")


# -----------------------
# Helpers
# -----------------------
def chunked(iterable: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def list_txt_keys(bucket: str, prefix: str = "") -> List[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".txt"):
                keys.append(obj["Key"])
    return keys


def download_texts(bucket: str, keys: List[str]) -> List[str]:
    texts = []
    for key in keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        texts.append(obj["Body"].read().decode("utf-8"))
    return texts


def upload_embeddings(bucket: str, keys: List[str], embeddings):
    for key, emb in zip(keys, embeddings):
        json_key = str(Path(key).with_suffix(".json"))
        payload = {
            "source_txt": key,
            "model": MODEL_NAME,
            "embedding": emb.tolist(),
        }

        # Uncomment when ready
        s3.put_object(
            Bucket=bucket,
            Key=json_key,
            Body=json.dumps(payload),
            ContentType="application/json",
        )


# -----------------------
# Main pipeline
# -----------------------
def main():
    print("Listing txt files...")
    txt_keys = list_txt_keys(BUCKET_NAME, PREFIX)
    print(f"Found {len(txt_keys)} files")

    print("Loading model on GPU...")
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        model_kwargs={
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={
            "padding_side": "left",
            "truncation": True,
            "max_length": 2048,  # <= IMPORTANT
        },
        device="cuda",
    )

    model.max_seq_length = 4096

    for i, key_chunk in tqdm(enumerate(chunked(txt_keys, KEY_CHUNK_SIZE), start=1)):
        print(f"\nProcessing chunk {i} ({len(key_chunk)} files)")

        texts = download_texts(BUCKET_NAME, key_chunk)

        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        upload_embeddings(BUCKET_NAME, key_chunk, embeddings)

        # ---- cleanup (VERY IMPORTANT) ----
        del texts, embeddings
        torch.cuda.empty_cache()

    print("Done!")


if __name__ == "__main__":
    main()
