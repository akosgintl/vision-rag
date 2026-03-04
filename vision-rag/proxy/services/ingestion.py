"""
Document ingestion service.

Handles PDF upload -> rasterization -> MinIO storage -> ColPali embedding -> Qdrant indexing.

Optimized for vLLM's continuous batching:
- Page embeddings are generated in a single batched /v1/embeddings call rather than
  one-at-a-time, letting vLLM's scheduler process all images concurrently using
  PagedAttention for efficient KV-cache sharing.
"""

import asyncio
import base64
import io
import time
import uuid
from functools import partial

import structlog

from proxy.config import settings
from proxy.services.backend import BackendCaller
from proxy.services.embedding_index import EmbeddingIndex
from proxy.services.job_tracker import JobTracker
from proxy.services.metadata import DocumentMetadata
from proxy.services.storage import MinioStorage

logger = structlog.get_logger()


class IngestionService:
    """Handles PDF ingestion pipeline."""

    def __init__(
        self,
        backend: BackendCaller,
        embedding_index: EmbeddingIndex,
        storage: MinioStorage,
        job_tracker: JobTracker | None = None,
        metadata_db: DocumentMetadata | None = None,
    ):
        self.backend = backend
        self.index = embedding_index
        self.storage = storage
        self.job_tracker = job_tracker
        self.metadata_db = metadata_db

    async def ingest_pdf(
        self,
        pdf_bytes: bytes,
        collection: str = "default",
        document_id: str | None = None,
        dpi: int = 300,
        metadata: dict | None = None,
        job_id: str | None = None,
    ) -> dict:
        """
        Full ingestion pipeline for a single PDF.

        1. Rasterize PDF pages to images
        2. Store images in MinIO
        3. Generate ColPali embeddings for each page
        4. Index embeddings in Qdrant
        5. Track progress via Redis job tracker

        Args:
            pdf_bytes: Raw PDF file bytes
            collection: Target collection name
            document_id: Optional custom document ID
            dpi: Rasterization DPI (default 300)
            metadata: Extra metadata to store
            job_id: Optional job ID for progress tracking

        Returns:
            dict with document_id, total_pages, status
        """
        doc_id = document_id or str(uuid.uuid4())
        effective_job_id = job_id or doc_id
        start_time = time.time()

        logger.info("ingestion_start", document_id=doc_id, collection=collection)

        # Register document in PostgreSQL
        if self.metadata_db:
            try:
                await self.metadata_db.register_document(
                    document_id=doc_id,
                    collection=collection,
                    dpi=dpi,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning("metadata_register_failed", document_id=doc_id, error=str(e))

        # Step 1: Rasterize PDF to page images (CPU-bound, use executor)
        loop = asyncio.get_running_loop()
        page_images = await loop.run_in_executor(None, partial(self._rasterize_pdf, pdf_bytes, dpi))
        total_pages = len(page_images)
        logger.info("pdf_rasterized", document_id=doc_id, pages=total_pages, dpi=dpi)

        if self.job_tracker:
            await self.job_tracker.update_progress(effective_job_id, 0, total_pages)

        # Step 2: Store images in MinIO
        try:
            stored = await self.storage.store_page_images(page_images, collection, doc_id)
            logger.info("images_stored", document_id=doc_id, stored=stored, total=total_pages)
        except Exception as e:
            logger.error("image_storage_failed", document_id=doc_id, error=str(e))
            if self.job_tracker:
                await self.job_tracker.fail_job(effective_job_id, f"Image storage failed: {e}")
            raise

        # Step 3: Batch-embed all pages in a single /v1/embeddings call
        logger.info("embedding_start", document_id=doc_id, pages=total_pages)
        all_embeddings = await self._embed_pages_batch(page_images)
        logger.info("embedding_complete", document_id=doc_id, pages=total_pages)

        # Step 4: Index each page in Qdrant
        indexed_count = 0
        for page_num, embeddings in enumerate(all_embeddings, start=1):
            if embeddings is None:
                logger.error("page_embedding_failed", document_id=doc_id, page=page_num)
                if self.job_tracker:
                    await self.job_tracker.update_progress(effective_job_id, page_num, total_pages)
                continue
            try:
                await self.index.index_page(
                    embeddings=embeddings,
                    document_id=doc_id,
                    page_number=page_num,
                    collection_name=collection,
                    metadata={
                        **(metadata or {}),
                        "dpi": dpi,
                        "total_pages": total_pages,
                    },
                )
                indexed_count += 1
            except Exception as e:
                logger.error(
                    "page_indexing_failed",
                    document_id=doc_id,
                    page=page_num,
                    error=str(e),
                )

            if self.job_tracker:
                await self.job_tracker.update_progress(effective_job_id, page_num, total_pages)

        elapsed = round(time.time() - start_time, 2)

        # Update metadata in PostgreSQL
        status = "completed" if indexed_count == total_pages else "partial"
        if self.metadata_db:
            try:
                await self.metadata_db.update_document(
                    document_id=doc_id,
                    page_count=total_pages,
                    indexed_pages=indexed_count,
                    status=status,
                )
            except Exception as e:
                logger.warning("metadata_update_failed", document_id=doc_id, error=str(e))

        # Mark job complete in Redis
        if self.job_tracker:
            await self.job_tracker.complete_job(effective_job_id, total_pages, indexed_count)

        logger.info(
            "ingestion_complete",
            document_id=doc_id,
            total_pages=total_pages,
            indexed=indexed_count,
            elapsed_seconds=elapsed,
        )

        return {
            "document_id": doc_id,
            "total_pages": total_pages,
            "indexed_pages": indexed_count,
            "collection": collection,
            "elapsed_seconds": elapsed,
            "status": status,
        }

    def _rasterize_pdf(self, pdf_bytes: bytes, dpi: int = 300) -> list[str]:
        """
        Convert PDF to list of base64-encoded PNG images.

        Requires poppler-utils system package.
        """
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png")

        page_images = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            page_images.append(b64)

        return page_images

    async def _embed_pages_batch(self, page_images: list[str]) -> list[list[list[float]] | None]:
        """
        Generate ColPali embeddings for all page images in a single batched
        /v1/embeddings call. vLLM processes the batch using continuous batching
        and PagedAttention for efficient GPU utilization.

        Returns a list of embeddings (one per page). On batch failure, falls
        back to per-page calls — failed pages return None.
        """
        input_items = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
            for b64 in page_images
        ]

        try:
            data = await self.backend.post(
                backend_name="colpali",
                url=f"{settings.colpali_url}/v1/embeddings",
                json={
                    "model": settings.colpali_model_id,
                    "input": input_items,
                },
            )
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.warning(
                "batch_embedding_failed_falling_back",
                error=str(e),
                pages=len(page_images),
            )
            # Fallback: embed one-by-one so partial failures don't lose all pages
            results: list[list[list[float]] | None] = []
            for b64 in page_images:
                try:
                    emb = await self._embed_page(b64)
                    results.append(emb)
                except Exception as page_e:
                    logger.error("page_embedding_fallback_failed", error=str(page_e))
                    results.append(None)
            return results

    async def _embed_page(self, image_b64: str) -> list[list[float]]:
        """Generate ColPali embeddings for a single page image (used as fallback)."""
        data = await self.backend.post(
            backend_name="colpali",
            url=f"{settings.colpali_url}/v1/embeddings",
            json={
                "model": settings.colpali_model_id,
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                ],
            },
        )
        return data["data"][0]["embedding"]
