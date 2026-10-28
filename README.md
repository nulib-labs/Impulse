# Now 100% sugar free

## New features

**Impulse manifests**: You can now build out meta-manifests for Impulse!
See `./tests/impulse_manifest/manifest.yaml` for details.

You can use that manifest with `uv run lpad add ./tests/impulse_manifest/manifest.yaml`

![Impulse Logo](logo.png)

Impulse is an easy-to-deploy all-in-one solution for extremely high-throughput
embarrassingly parallel data extraction built for digitization, indexing,
and distribution through IIIF manifests.

Running on Northwestern University's Quest supercomputing cluster, we observed
hundreds of pages per hour. (This is while limiting ourselves to 50 concurrent
jobs!) This could easily hit thousands of pages per hour.

This code is very easily extensible to new models/paradigms. Just edit the
`impulse.tasks.DocumentExtraction` class!

## Running Surya-2 OCR on a DGX Spark (GB10)

```sh
scripts/run_vllm_spark.sh          # start surya-ocr-2 on vLLM, tuned for GB10
scripts/run_worker.sh              # FireWorks worker; attaches to it via AsyncOpenAI
uv run python scripts/bench_surya_vllm.py page*.jpg --repeat 4   # measure pages/s
```

Tuning knobs are environment variables documented at the top of each script
(`VLLM_MAX_NUM_SEQS`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MTP_TOKENS`,
`SURYA_ASYNC_CONCURRENCY`, `SURYA_PAGE_WORKERS`). The async client lives in
`tasks/common/surya_async.py` and can be awaited directly:
`await make_async_manager().backend.agenerate(batch_input_items)`.

This project would not be possible without the work of other amazing open-
source projects.

- [Fireworks by materialsproject](https://materialsproject.github.io/fireworks/)
- [Surya by DataLab](https://github.com/datalab-to/surya)
- [Sentence-Transformers](https://sbert.net/)
