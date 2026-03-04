"""Retrieval router — proxies to ColPali vLLM for embedding search."""

import time

import structlog
from fastapi import APIRouter, HTTPException, Request

from proxy.config import settings
from proxy.models.requests import (
    RetrievalBatchIndexRequest,
    RetrievalIndexRequest,
    RetrievalSearchRequest,
)
from proxy.models.responses import (
    RetrievalIndexResponse,
    RetrievalResult,
    RetrievalSearchResponse,
)
from proxy.services.backend import BackendError
from proxy.services.circuit_breaker import CircuitOpenError

logger = structlog.get_logger()
router = APIRouter(prefix="/v1/retrieve", tags=["retrieval"])


@router.post("/search", response_model=RetrievalSearchResponse)
async def search(req: RetrievalSearchRequest, request: Request):
    """
    Search for relevant document pages using ColPali.

    1. Encode the text query via ColPali embeddings endpoint
    2. Search Qdrant for similar page embeddings (MaxSim scoring)
    3. Return ranked results
    """
    start = time.time()
    backend = request.app.state.backend
    index = request.app.state.embedding_index

    try:
        data = await backend.post(
            backend_name="colpali",
            url=f"{settings.colpali_url}/v1/embeddings",
            json={
                "model": settings.colpali_model_id,
                "input": [{"type": "text", "text": req.query}],
            },
        )
        query_embeddings = data["data"][0]["embedding"]
    except CircuitOpenError as e:
        raise HTTPException(
            503, f"ColPali service unavailable (circuit open). Retry after {e.retry_after:.0f}s."
        ) from e
    except BackendError as e:
        raise HTTPException(502, f"ColPali embedding failed: {e.message}") from e

    # Search Qdrant
    results = await index.search(
        query_embeddings=query_embeddings,
        collection_name=req.collection,
        top_k=req.top_k,
    )

    latency = round((time.time() - start) * 1000, 2)
    return RetrievalSearchResponse(
        results=[
            RetrievalResult(
                document_id=r["document_id"],
                page_number=r["page_number"],
                score=round(r["score"], 4),
                collection=r["collection"],
            )
            for r in results
        ],
        query=req.query,
        latency_ms=latency,
    )


@router.post("/index", response_model=RetrievalIndexResponse)
async def index_page(req: RetrievalIndexRequest, request: Request):
    """Index a single document page by generating and storing ColPali embeddings."""
    backend = request.app.state.backend
    index = request.app.state.embedding_index

    try:
        data = await backend.post(
            backend_name="colpali",
            url=f"{settings.colpali_url}/v1/embeddings",
            json={
                "model": settings.colpali_model_id,
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{req.image_base64}"},
                    }
                ],
            },
        )
        embeddings = data["data"][0]["embedding"]
    except CircuitOpenError as e:
        raise HTTPException(503, f"ColPali service unavailable. Retry after {e.retry_after:.0f}s.") from e
    except BackendError as e:
        raise HTTPException(502, f"ColPali embedding failed: {e.message}") from e

    # Store in Qdrant
    await index.index_page(
        embeddings=embeddings,
        document_id=req.document_id,
        page_number=req.page_number,
        collection_name=req.collection,
        metadata=req.metadata,
    )

    return RetrievalIndexResponse(
        document_id=req.document_id,
        page_number=req.page_number,
        status="indexed",
        embedding_dim=len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else None,
    )


@router.post("/index/batch", response_model=list[RetrievalIndexResponse])
async def index_batch(req: RetrievalBatchIndexRequest, request: Request):
    """
    Index multiple pages with batched ColPali embeddings.

    All page images are embedded in a single /v1/embeddings call, letting
    vLLM's continuous batching process them concurrently. Qdrant indexing
    follows per-page with individual error handling.
    """
    backend = request.app.state.backend
    index = request.app.state.embedding_index

    # Batch-embed all page images in a single ColPali call
    input_items = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{page.image_base64}"},
        }
        for page in req.pages
    ]

    try:
        data = await backend.post(
            backend_name="colpali",
            url=f"{settings.colpali_url}/v1/embeddings",
            json={
                "model": settings.colpali_model_id,
                "input": input_items,
            },
        )
        all_embeddings = [item["embedding"] for item in data["data"]]
    except CircuitOpenError as e:
        raise HTTPException(503, f"ColPali service unavailable. Retry after {e.retry_after:.0f}s.") from e
    except BackendError as e:
        raise HTTPException(502, f"ColPali batch embedding failed: {e.message}") from e

    # Index each page in Qdrant
    results = []
    for page, embeddings in zip(req.pages, all_embeddings, strict=True):
        try:
            await index.index_page(
                embeddings=embeddings,
                document_id=page.document_id,
                page_number=page.page_number,
                collection_name=page.collection,
                metadata=page.metadata,
            )
            results.append(
                RetrievalIndexResponse(
                    document_id=page.document_id,
                    page_number=page.page_number,
                    status="indexed",
                    embedding_dim=len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else None,
                )
            )
        except Exception as e:
            logger.error("batch_index_page_failed", document_id=page.document_id, error=str(e))
            results.append(
                RetrievalIndexResponse(
                    document_id=page.document_id,
                    page_number=page.page_number,
                    status=f"failed: {e}",
                )
            )

    return results
