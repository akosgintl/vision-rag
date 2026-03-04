"""
Celery task definitions for async document processing.

Tasks run in a separate Celery worker process, communicating
status via the Redis-backed JobTracker.
"""

import asyncio

import httpx
import structlog
from celery import Celery

from proxy.config import settings
from proxy.services.backend import BackendCaller
from proxy.services.circuit_breaker import CircuitBreakerRegistry
from proxy.services.embedding_index import EmbeddingIndex
from proxy.services.ingestion import IngestionService
from proxy.services.job_tracker import JobTracker
from proxy.services.metadata import DocumentMetadata
from proxy.services.storage import MinioStorage

logger = structlog.get_logger()

celery_app = Celery(
    "vision_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _build_ingestion_service() -> tuple[IngestionService, httpx.AsyncClient, JobTracker, DocumentMetadata | None]:
    """Build services needed for ingestion within the Celery worker."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
    )
    breakers = CircuitBreakerRegistry(
        failure_threshold=settings.circuit_failure_threshold,
        recovery_timeout=settings.circuit_recovery_timeout,
    )
    backend = BackendCaller(client=client, circuit_breakers=breakers)
    index = EmbeddingIndex()
    storage = MinioStorage()
    job_tracker = JobTracker()

    try:
        metadata_db = DocumentMetadata()
    except Exception:
        metadata_db = None

    service = IngestionService(
        backend=backend,
        embedding_index=index,
        storage=storage,
        job_tracker=job_tracker,
        metadata_db=metadata_db,
    )
    return service, client, job_tracker, metadata_db


@celery_app.task(name="vision_rag.ingest_document", bind=True, max_retries=2)
def ingest_document(
    self,
    job_id: str,
    pdf_bytes_hex: str,
    collection: str = "default",
    dpi: int = 300,
    metadata: dict | None = None,
):
    """
    Celery task: ingest a PDF document.

    pdf_bytes are passed as hex-encoded string since Celery uses JSON serialization.
    """
    pdf_bytes = bytes.fromhex(pdf_bytes_hex)

    async def _run():
        service, client, job_tracker, metadata_db = _build_ingestion_service()
        try:
            result = await service.ingest_pdf(
                pdf_bytes=pdf_bytes,
                collection=collection,
                document_id=job_id,
                dpi=dpi,
                metadata=metadata,
                job_id=job_id,
            )
            return result
        except Exception as e:
            await job_tracker.fail_job(job_id, str(e))
            logger.error("celery_ingest_failed", job_id=job_id, error=str(e))
            raise
        finally:
            await client.aclose()
            await job_tracker.close()
            if metadata_db:
                metadata_db.close()

    return asyncio.run(_run())
