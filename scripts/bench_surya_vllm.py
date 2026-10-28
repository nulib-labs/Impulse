"""Benchmark / smoke-test the surya-ocr-2 vLLM server through the async client.

Sends one full-page OCR request (the same request DocumentExtractionTask makes)
per image, all at once, through ``tasks.common.surya_async.AsyncVllmBackend``,
and reports pages/s and output tokens/s. Use it to tune
``SURYA_ASYNC_CONCURRENCY`` and the server flags in scripts/run_vllm_spark.sh.

Run with:
    SURYA_INFERENCE_URL=http://127.0.0.1:8000/v1 \
        uv run python scripts/bench_surya_vllm.py page1.jpg page2.jpg ... [--concurrency 128] [--repeat 4]
"""

import asyncio
import importlib.util
import time
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from surya.inference.schema import PROMPT_TYPE_HIGH_ACCURACY_BBOX, BatchInputItem
from surya.settings import settings

# Load the backend by path: importing the ``tasks`` package pulls in my_tasks,
# which requires S3/AWS env vars this script doesn't need.
_spec = importlib.util.spec_from_file_location(
    "surya_async",
    Path(__file__).resolve().parent.parent / "tasks" / "common" / "surya_async.py",
)
surya_async = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(surya_async)


async def main(image_paths: list[str], concurrency: int, repeat: int) -> None:
    images = [Image.open(p).convert("RGB") for p in image_paths] * repeat
    batch = [
        BatchInputItem(
            image=img,
            prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
            max_tokens=settings.SURYA_MAX_TOKENS_FULL_PAGE,
        )
        for img in images
    ]
    backend = surya_async.AsyncVllmBackend(max_concurrency=concurrency)
    try:
        # Warm-up: the first request after server boot pays JIT/compile cost.
        await backend.agenerate(batch[:1])

        start = time.perf_counter()
        outputs = await backend.agenerate(batch)
        elapsed = time.perf_counter() - start
    finally:
        backend.stop()

    tokens = sum(o.token_count for o in outputs)
    errors = sum(o.error for o in outputs)
    print(f"pages:        {len(outputs)} ({errors} errors) at concurrency {concurrency}")
    print(f"wall time:    {elapsed:.1f} s")
    print(f"throughput:   {len(outputs) / elapsed:.2f} pages/s, {tokens / elapsed:.0f} output tok/s")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", help="page images to OCR")
    parser.add_argument("--concurrency", type=int, default=surya_async.SURYA_ASYNC_CONCURRENCY)
    parser.add_argument("--repeat", type=int, default=1, help="send each image N times")
    args = parser.parse_args()
    asyncio.run(main(args.images, args.concurrency, args.repeat))
