"""Generation router — proxies to Qwen2.5-7B for text generation and summarization."""

import time

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from proxy.config import settings
from proxy.models.requests import GenerateChatRequest, GenerateSummarizeRequest
from proxy.models.responses import GenerateChatResponse, GenerateSummarizeResponse
from proxy.prompts import SummarizationPrompts
from proxy.services.backend import BackendError
from proxy.services.circuit_breaker import CircuitOpenError

logger = structlog.get_logger()
router = APIRouter(prefix="/v1/generate", tags=["generation"])


@router.post("/chat", response_model=GenerateChatResponse)
async def chat(req: GenerateChatRequest, request: Request):
    """
    Chat completion via Qwen2.5-7B.

    Accepts OpenAI-compatible messages format.
    Supports streaming when stream=True.
    """
    backend = request.app.state.backend
    start = time.time()

    payload = {
        "model": settings.qwen25_model_id,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }

    try:
        if req.stream:
            return await _stream_response(backend, payload)

        data = await backend.post(
            backend_name="qwen25",
            url=f"{settings.qwen25_url}/v1/chat/completions",
            json=payload,
        )

        content = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        finish_reason = data["choices"][0].get("finish_reason", "stop")
        latency_ms = round((time.time() - start) * 1000, 2)

        return GenerateChatResponse(
            content=content,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    except CircuitOpenError as e:
        raise HTTPException(503, f"Qwen2.5 service unavailable. Retry after {e.retry_after:.0f}s.") from e
    except BackendError as e:
        raise HTTPException(502, f"Qwen2.5 chat failed: {e.message}") from e


@router.post("/summarize", response_model=GenerateSummarizeResponse)
async def summarize(req: GenerateSummarizeRequest, request: Request):
    """Summarize extracted document content via Qwen2.5-7B."""
    backend = request.app.state.backend
    start = time.time()

    try:
        data = await backend.post(
            backend_name="qwen25",
            url=f"{settings.qwen25_url}/v1/chat/completions",
            json={
                "model": settings.qwen25_model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": SummarizationPrompts.format_system(req.style),
                    },
                    {
                        "role": "user",
                        "content": SummarizationPrompts.format_user(req.content),
                    },
                ],
                "max_tokens": req.max_tokens,
                "temperature": 0.2,
            },
        )

        summary = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        latency_ms = round((time.time() - start) * 1000, 2)

        return GenerateSummarizeResponse(
            summary=summary,
            style=req.style,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    except CircuitOpenError as e:
        raise HTTPException(503, f"Qwen2.5 service unavailable. Retry after {e.retry_after:.0f}s.") from e
    except BackendError as e:
        raise HTTPException(502, f"Qwen2.5 summarization failed: {e.message}") from e


async def _stream_response(backend, payload: dict):
    """Stream chat completions via SSE with error handling."""

    async def generate():
        try:
            async for chunk in backend.stream(
                backend_name="qwen25",
                url=f"{settings.qwen25_url}/v1/chat/completions",
                json=payload,
            ):
                yield chunk
        except BackendError as e:
            logger.error("stream_backend_error", error=str(e))
            yield f'data: {{"error": "{e.message}"}}\n\n'
        except Exception as e:
            logger.error("stream_unexpected_error", error=str(e))
            yield 'data: {"error": "Stream interrupted"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
