"""Extraction router — proxies to Qwen3-VL for visual document understanding.

Batch extraction fires all requests concurrently — vLLM's server-side continuous
batching scheduler processes them optimally using PagedAttention, so no proxy-side
semaphore is needed.
"""

import asyncio
import time

import structlog
from fastapi import APIRouter, HTTPException, Request

from proxy.config import settings
from proxy.models.requests import ExtractBatchRequest, ExtractPageRequest
from proxy.models.responses import ExtractBatchResponse, ExtractionResult
from proxy.services.backend import BackendCaller, BackendError
from proxy.services.circuit_breaker import CircuitOpenError

logger = structlog.get_logger()
router = APIRouter(prefix="/v1/extract", tags=["extraction"])


async def _extract_single(backend: BackendCaller, req: ExtractPageRequest) -> ExtractionResult:
    """Call Qwen3-VL to extract structured data from a page image."""
    start = time.time()

    data = await backend.post(
        backend_name="qwen3vl",
        url=f"{settings.qwen3vl_url}/v1/chat/completions",
        json={
            "model": settings.qwen3vl_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{req.image_base64}"},
                        },
                        {
                            "type": "text",
                            "text": req.prompt,
                        },
                    ],
                }
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        },
    )

    content = data["choices"][0]["message"]["content"]
    tokens_used = data.get("usage", {}).get("total_tokens", 0)
    latency_ms = round((time.time() - start) * 1000, 2)

    return ExtractionResult(
        content=content,
        format=req.output_format,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )


@router.post("/page", response_model=ExtractionResult)
async def extract_page(req: ExtractPageRequest, request: Request):
    """Extract structured data from a single document page via Qwen3-VL."""
    backend = request.app.state.backend
    try:
        return await _extract_single(backend, req)
    except CircuitOpenError as e:
        raise HTTPException(503, f"Qwen3-VL service unavailable. Retry after {e.retry_after:.0f}s.") from e
    except BackendError as e:
        raise HTTPException(502, f"Qwen3-VL extraction failed: {e.message}") from e


@router.post("/batch", response_model=ExtractBatchResponse)
async def extract_batch(req: ExtractBatchRequest, request: Request):
    """
    Extract from multiple pages concurrently.

    All requests are fired simultaneously — vLLM's server-side continuous
    batching scheduler processes them using PagedAttention for maximum GPU
    utilization without needing a proxy-side concurrency limit.
    """
    backend = request.app.state.backend
    start = time.time()

    if req.concurrent:
        # Fire all concurrently — vLLM handles GPU scheduling internally
        tasks = [_extract_single(backend, page) for page in req.pages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, (BackendError, CircuitOpenError)):
                logger.error("batch_extraction_failed", page=i, error=str(result))
                processed.append(
                    ExtractionResult(
                        content=f"[Extraction failed: {result}]",
                        format=req.pages[i].output_format,
                        tokens_used=0,
                        latency_ms=0,
                    )
                )
            elif isinstance(result, Exception):
                logger.error("batch_extraction_failed", page=i, error=str(result))
                processed.append(
                    ExtractionResult(
                        content="[Extraction failed: unexpected error]",
                        format=req.pages[i].output_format,
                        tokens_used=0,
                        latency_ms=0,
                    )
                )
            else:
                processed.append(result)
    else:
        processed = []
        for page in req.pages:
            try:
                result = await _extract_single(backend, page)
                processed.append(result)
            except (BackendError, CircuitOpenError) as e:
                processed.append(
                    ExtractionResult(
                        content=f"[Extraction failed: {e}]",
                        format=page.output_format,
                        tokens_used=0,
                        latency_ms=0,
                    )
                )

    total_tokens = sum(r.tokens_used for r in processed)
    total_latency = round((time.time() - start) * 1000, 2)

    return ExtractBatchResponse(
        results=processed,
        total_tokens=total_tokens,
        total_latency_ms=total_latency,
    )
