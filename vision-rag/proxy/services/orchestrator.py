"""
Pipeline Orchestrator — chains all three models.

Flow: ColPali (retrieve) -> Qwen3-VL (extract) -> Qwen2.5-7B (generate)

Optimized for vLLM's continuous batching and PagedAttention:
- Image fetches are parallelized via asyncio.gather.
- Extraction requests are fired concurrently without a proxy-side semaphore —
  vLLM's server-side scheduler batches them internally using PagedAttention,
  overlapping prefill and decode across sequences for maximum GPU utilization.
"""

import asyncio
import time

import structlog

from proxy.config import settings
from proxy.models.responses import (
    PipelineQueryResponse,
    SourceReference,
    TokenUsage,
)
from proxy.prompts import ExtractionPrompts, GenerationPrompts
from proxy.services.backend import BackendCaller
from proxy.services.embedding_index import EmbeddingIndex
from proxy.services.storage import MinioStorage

logger = structlog.get_logger()


class PipelineOrchestrator:
    """Orchestrates the full document understanding pipeline."""

    def __init__(
        self,
        backend: BackendCaller,
        embedding_index: EmbeddingIndex,
        storage: MinioStorage,
    ):
        self.backend = backend
        self.index = embedding_index
        self.storage = storage

    async def query(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        include_extractions: bool = True,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> PipelineQueryResponse:
        """
        Full pipeline execution:
          1. Encode query via ColPali -> get query embeddings
          2. Search Qdrant for top-K similar pages
          3. Retrieve page images from storage (concurrent)
          4. Extract structured data via Qwen3-VL (concurrent — vLLM batches internally)
          5. Generate answer via Qwen2.5-7B
        """
        pipeline_start = time.time()
        tokens = TokenUsage()

        # Step 1: Encode query via ColPali
        logger.info("pipeline_step", step="retrieve", query=query[:100])
        query_embeddings, retrieval_tokens = await self._encode_query(query)
        tokens.retrieval_tokens = retrieval_tokens

        # Step 2: Search Qdrant
        search_results = await self.index.search(
            query_embeddings=query_embeddings,
            collection_name=collection,
            top_k=top_k,
        )

        if not search_results:
            return PipelineQueryResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                extractions=None,
                latency_ms=round((time.time() - pipeline_start) * 1000, 2),
                tokens=tokens,
            )

        sources = [
            SourceReference(
                document_id=r["document_id"],
                page_number=r["page_number"],
                relevance_score=round(r["score"], 4),
                collection=r["collection"],
            )
            for r in search_results
        ]

        # Step 3: Retrieve page images (concurrent fetches)
        page_images = await self._fetch_page_images(search_results)

        # Step 4: Extract via Qwen3-VL (concurrent — vLLM batches internally)
        logger.info("pipeline_step", step="extract", pages=len(page_images))
        extractions, extraction_tokens = await self._extract_pages_batch(page_images, query)
        tokens.extraction_tokens = extraction_tokens

        # Step 5: Generate answer via Qwen2.5-7B
        logger.info("pipeline_step", step="generate")
        context = self._build_context(extractions, sources)
        answer, gen_tokens = await self._generate_answer(query, context, max_tokens, temperature)
        tokens.generation_tokens = gen_tokens

        latency_ms = round((time.time() - pipeline_start) * 1000, 2)
        logger.info("pipeline_complete", latency_ms=latency_ms, total_tokens=tokens.total)

        return PipelineQueryResponse(
            answer=answer,
            sources=sources,
            extractions=extractions if include_extractions else None,
            latency_ms=latency_ms,
            tokens=tokens,
        )

    # ─── Private helpers ─────────────────────────────────────

    async def _encode_query(self, query: str) -> tuple[list[list[float]], int]:
        """Encode a text query into multi-vector embeddings via ColPali."""
        data = await self.backend.post(
            backend_name="colpali",
            url=f"{settings.colpali_url}/v1/embeddings",
            json={
                "model": settings.colpali_model_id,
                "input": [{"type": "text", "text": query}],
            },
        )
        embeddings = data["data"][0]["embedding"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return embeddings, tokens_used

    async def _fetch_page_images(self, search_results: list[dict]) -> list[str]:
        """
        Fetch page images from MinIO concurrently via asyncio.gather.

        Returns list of base64-encoded images. Failed fetches return empty
        strings and are logged — downstream extraction handles missing images.
        """

        async def _fetch_one(result: dict) -> str:
            try:
                return await self.storage.fetch_page_image(
                    result.get("collection", "default"),
                    result["document_id"],
                    result["page_number"],
                )
            except Exception as e:
                logger.error(
                    "page_image_fetch_failed",
                    document_id=result["document_id"],
                    page=result["page_number"],
                    error=str(e),
                )
                return ""

        return list(await asyncio.gather(*[_fetch_one(r) for r in search_results]))

    async def _extract_pages_batch(self, page_images: list[str], query: str) -> tuple[list[str], int]:
        """
        Extract structured data from all pages concurrently via Qwen3-VL.

        All requests are fired simultaneously without a proxy-side semaphore.
        vLLM's server-side continuous batching scheduler picks them all up and
        processes them optimally using PagedAttention — overlapping prefill
        and decode across sequences to maximize GPU utilization.
        """
        valid_entries = [(i, img) for i, img in enumerate(page_images) if img]

        if not valid_entries:
            return ["[Image unavailable]"] * len(page_images), 0

        # Fire all extraction requests concurrently — let vLLM batch
        tasks = [self._extract_page(img, query) for _, img in valid_entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Reassemble into page-ordered list
        extractions: list[str] = []
        total_tokens = 0
        result_idx = 0
        for i, img in enumerate(page_images):
            if not img:
                extractions.append(f"[Image unavailable for page {i + 1}]")
                continue
            result = results[result_idx]
            result_idx += 1
            if isinstance(result, Exception):
                logger.error("extraction_failed", page=i, error=str(result))
                extractions.append(f"[Extraction failed for page {i + 1}]")
            else:
                content, tok = result
                extractions.append(content)
                total_tokens += tok

        return extractions, total_tokens

    async def _extract_page(self, image_b64: str, query: str) -> tuple[str, int]:
        """Extract structured data from a page image via Qwen3-VL."""
        data = await self.backend.post(
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
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                            {
                                "type": "text",
                                "text": ExtractionPrompts.format_query_contextual(query),
                            },
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
            },
        )
        content = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens_used

    async def _generate_answer(
        self,
        query: str,
        context: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        """Generate a natural-language answer via Qwen2.5-7B."""
        data = await self.backend.post(
            backend_name="qwen25",
            url=f"{settings.qwen25_url}/v1/chat/completions",
            json={
                "model": settings.qwen25_model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": GenerationPrompts.SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": GenerationPrompts.format_user(context, query),
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        answer = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used

    def _build_context(self, extractions: list[str], sources: list[SourceReference]) -> str:
        """Build context string from extractions with page references."""
        parts = []
        for _i, (extraction, source) in enumerate(zip(extractions, sources, strict=False)):
            parts.append(
                f"[Page {source.page_number} from {source.document_id} "
                f"(relevance: {source.relevance_score:.3f})]\n{extraction}"
            )
        return "\n\n---\n\n".join(parts)
