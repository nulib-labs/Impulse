import json
from pathlib import Path
from typing import List

import boto3
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
BUCKET_NAME = "nu-impulse-production"
PREFIX = "P0491_35556036056489"  # e.g. "corpus/texts/"
BATCH_SIZE = 128  # tune based on GPU memory
AWS_PROFILE = "impulse"
MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
session = boto3.Session(profile_name=AWS_PROFILE)
s3 = session.client("s3")


# -----------------------
# S3 helpers
# -----------------------
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

    print(texts)
    return texts


def upload_embeddings(bucket: str, keys: List[str], embeddings):
    for key, emb in zip(keys, embeddings):
        json_key = str(Path(key).with_suffix(".json"))

        payload = {
            "source_txt": key,
            "model": MODEL_NAME,
            "embedding": emb.tolist(),
        }
        print(f"Now uploading payload to {key}")
        # s3.put_object(
        #     Bucket=bucket,
        #     Key=json_key,
        #     Body=json.dumps(payload),
        #     ContentType="application/json",
        # )


# -----------------------
# Main pipeline
# -----------------------
def main():
    print("Listing txt files...")
    txt_keys = list_txt_keys(BUCKET_NAME, PREFIX)
    print(f"Found {len(txt_keys)} files")

    print("Downloading texts...")
    texts = download_texts(BUCKET_NAME, txt_keys)

    print("Loading model on GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)

    print("Encoding texts...")
    attn_implementation = "eager"  # Or "flash_attention_2"
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        model_kwargs={
            "attn_implementation": attn_implementation,
            "torch_dtype": "bfloat16",
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Uploading embeddings...")
    upload_embeddings(BUCKET_NAME, txt_keys, embeddings)

    print("Done!")


if __name__ == "__main__":
    main()
