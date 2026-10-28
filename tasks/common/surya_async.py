"""AsyncOpenAI backend for Surya-2 against a long-running vLLM server.

Surya's stock vllm backend drives the server with the *sync* OpenAI client and a
thread pool capped at ``SURYA_INFERENCE_PARALLEL`` (default 8), and each
``generate`` call blocks until its slowest request finishes. On a DGX Spark the
~0.7B surya-ocr-2 model is memory-bandwidth bound, so throughput comes from
keeping hundreds of sequences in flight, which 8 threads never do.

This backend sends the same chat-completions requests (same prompts, image
scaling, sampling params and repeat/err retry policy as
``surya.inference.backends.openai_client``) through one ``AsyncOpenAI`` client
running on a dedicated event-loop thread. Every caller in the process shares one
connection pool and one in-flight limit (``SURYA_ASYNC_CONCURRENCY``).

Usage::

    manager = make_async_manager()            # drop-in for SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)
    recognition_predictor = RecognitionPredictor(manager)

    # From async code (any event loop):
    outputs = await manager.backend.agenerate(batch_input_items)
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import List, Optional

import httpx
from loguru import logger
from openai import AsyncOpenAI
from surya.inference import SuryaInferenceManager
from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.backends.openai_client import (
    _build_messages,
    _mean_token_prob,
    _should_retry,
)
from surya.inference.backends.vllm import VllmBackend
from surya.inference.prompts import PROMPT_MAPPING
from surya.inference.schema import BatchInputItem, BatchOutputItem, GenerationResult
from surya.inference.util import scale_to_fit
from surya.settings import settings

# Max requests in flight to vLLM from this process. Keep it at or below the
# server's --max-num-seqs (infra/spark/docker-compose.yml) divided by the number
# of worker processes sharing the server.
SURYA_ASYNC_CONCURRENCY = int(os.environ.get("SURYA_ASYNC_CONCURRENCY", "128"))


class AsyncVllmBackend(Backend):
    name = "vllm-async"

    def __init__(
        self,
        max_concurrency: int = SURYA_ASYNC_CONCURRENCY,
        max_tokens_default: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_retries: int = 3,
    ):
        self.max_concurrency = max_concurrency
        self.max_tokens_default = max_tokens_default
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries

        self.handle: Optional[ServerHandle] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[AsyncOpenAI] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = threading.Lock()

    def start(self) -> ServerHandle:
        with self._lock:
            if self.handle is not None:
                return self.handle
            # Reuse surya's attach logic: /health probe + served-model-name check
            # (and docker autostart if SURYA_INFERENCE_URL is unset).
            handle = VllmBackend().start()

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="surya-async-openai", daemon=True
            )
            self._thread.start()
            asyncio.run_coroutine_threadsafe(
                self._open_client(handle.base_url), self._loop
            ).result()
            self.handle = handle
            logger.info(
                f"Surya async client attached to {handle.base_url} "
                f"(max_concurrency={self.max_concurrency})"
            )
            return handle

    async def _open_client(self, base_url: str) -> None:
        timeout = settings.SURYA_INFERENCE_TIMEOUT_SECONDS
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self.max_concurrency,
                max_keepalive_connections=self.max_concurrency,
            ),
            timeout=timeout,
        )
        self._client = AsyncOpenAI(
            api_key=settings.VLLM_API_KEY,
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    def stop(self) -> None:
        with self._lock:
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._loop
                ).result()
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()
                self._loop.close()
            self.handle = None
            self._loop = None
            self._thread = None
            self._client = None
            self._semaphore = None

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        """Blocking entry point used by surya's predictors. Thread-safe."""
        if not batch:
            return []
        if self.handle is None:
            self.start()
        return asyncio.run_coroutine_threadsafe(
            self._generate_batch(batch), self._loop
        ).result()

    async def agenerate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        """Awaitable entry point, usable from any event loop."""
        if not batch:
            return []
        if self.handle is None:
            await asyncio.to_thread(self.start)
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(self._generate_batch(batch), self._loop)
        )

    async def _generate_batch(
        self, batch: List[BatchInputItem]
    ) -> List[BatchOutputItem]:
        return list(await asyncio.gather(*(self._process(item) for item in batch)))

    async def _process(self, item: BatchInputItem) -> BatchOutputItem:
        # Mirrors surya.inference.backends.openai_client.chat_completions_batch.
        result = await self._generate_one(item, self.temperature, self.top_p)
        retries = 0
        while _should_retry(result, retries, self.max_retries):
            if result.error:
                await asyncio.sleep(1.5 * (retries + 1))
            retry_temp = min(self.temperature + 0.2 * (retries + 1), 0.8)
            retry_top_p = 0.95 if not result.error else self.top_p
            result = await self._generate_one(item, retry_temp, retry_top_p)
            retries += 1
        return BatchOutputItem(
            raw=result.raw,
            token_count=result.token_count,
            error=result.error,
            mean_token_prob=result.mean_token_prob,
            logprobs=result.logprobs,
            metadata=item.metadata,
        )

    async def _generate_one(
        self, item: BatchInputItem, temperature: float, top_p: float
    ) -> GenerationResult:
        prompt = item.prompt or PROMPT_MAPPING[item.prompt_type]
        request_logprobs = item.request_logprobs or settings.SURYA_INFERENCE_LOGPROBS

        kwargs = dict(
            model=self.handle.model_name,
            messages=None,
            max_tokens=item.max_tokens or self.max_tokens_default,
            temperature=temperature,
            top_p=top_p,
        )
        if request_logprobs:
            kwargs["logprobs"] = True
        if item.guided_json is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": item.guided_json,
                    "strict": True,
                },
            }
        if item.guided_regex is not None:
            # vLLM >= 0.12 replaced the top-level guided_* params.
            kwargs["extra_body"] = {"structured_outputs": {"regex": item.guided_regex}}

        async with self._semaphore:
            try:
                # Resize + PNG-encode off the event loop; full pages take tens of ms.
                kwargs["messages"] = await asyncio.to_thread(
                    lambda: _build_messages(scale_to_fit(item.image), prompt)
                )
                completion = await self._client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.warning(f"Inference error: {e}")
                return GenerationResult(raw="", token_count=0, error=True)

        choice = completion.choices[0]
        mean_p = None
        logprobs_content = None
        if request_logprobs and choice.logprobs is not None and choice.logprobs.content:
            logprobs_content = [c.model_dump() for c in choice.logprobs.content]
            mean_p = _mean_token_prob(choice.logprobs.content)
        return GenerationResult(
            raw=choice.message.content or "",
            token_count=completion.usage.completion_tokens if completion.usage else 0,
            error=False,
            mean_token_prob=mean_p,
            logprobs=logprobs_content,
        )


def make_async_manager(
    max_concurrency: int = SURYA_ASYNC_CONCURRENCY,
) -> SuryaInferenceManager:
    """A SuryaInferenceManager whose backend is ``AsyncVllmBackend``."""
    manager = SuryaInferenceManager(method="vllm")
    manager.method = AsyncVllmBackend.name
    manager.backend = AsyncVllmBackend(max_concurrency=max_concurrency)
    return manager
