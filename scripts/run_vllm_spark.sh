#!/usr/bin/env bash
# Launch the surya-ocr-2 vLLM server on an NVIDIA DGX Spark (GB10, aarch64,
# 128 GB unified memory) for scripts/run_worker.sh to attach to.
#
# Usage:
#   scripts/run_vllm_spark.sh                      # start (or replace) the server
#   scripts/run_vllm_spark.sh --enforce-eager      # extra args go to vllm serve
#   VLLM_MTP_TOKENS=0 scripts/run_vllm_spark.sh    # disable MTP speculation
#   docker logs -f surya-vllm / docker rm -f surya-vllm
#
# Tuning rationale (GB10):
#   - surya-ocr-2 is ~0.7B params (~1.4 GB bf16), so decode is bound by the
#     ~273 GB/s LPDDR5x bandwidth, not capacity. Throughput comes from batching
#     many sequences per step: MAX_NUM_SEQS=256 vs Surya's 4090 default of 32.
#   - GPU and CPU share one 128 GB pool and vLLM's 0.9 default grabs most of
#     it. 0.5 (~60 GB) is far more KV cache than 256 OCR sequences need and
#     leaves room for the worker's decoded page images.
#   - MAX_MODEL_LEN covers full-page OCR: up to ~8k image tokens (3072x2048 at
#     28 px/token) + 12288 output tokens (SURYA_MAX_TOKENS_FULL_PAGE). Surya's
#     18000 default rejects the largest pages.
#   - MAX_BATCHED_TOKENS lets ~4 full-page image prefills share one step.
#   - Bound to 127.0.0.1: the OpenAI endpoint has no auth. Set VLLM_BIND=0.0.0.0
#     only behind a firewall if workers run on other hosts.

set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
NAME="${VLLM_CONTAINER_NAME:-surya-vllm}"
MODEL="${SURYA_MODEL_CHECKPOINT:-datalab-to/surya-ocr-2}"
BIND="${VLLM_BIND:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"
MTP_TOKENS="${VLLM_MTP_TOKENS:-2}"

ARGS=(
    --model "$MODEL"
    --served-model-name "$MODEL"
    --dtype bfloat16
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-prefix-caching
    --limit-mm-per-prompt '{"image": 1}'
    --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
)
if [ "$MTP_TOKENS" -gt 0 ]; then
    ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS}}")
fi
ARGS+=("$@")

mkdir -p "$HF_CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -p "${BIND}:${PORT}:8000" \
    "$IMAGE" \
    "${ARGS[@]}"

HOST="${BIND/#0.0.0.0/127.0.0.1}"
echo "Waiting for ${NAME} on http://${HOST}:${PORT}/health (first start downloads the model and compiles CUDA graphs)..."
for _ in $(seq 1 120); do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
        echo "Ready: SURYA_INFERENCE_URL=http://${HOST}:${PORT}/v1"
        exit 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
        docker logs --tail 50 "$NAME"
        echo "Container ${NAME} exited." >&2
        exit 1
    fi
    sleep 5
done
echo "Timed out after 10 min; check: docker logs -f ${NAME}" >&2
exit 1
