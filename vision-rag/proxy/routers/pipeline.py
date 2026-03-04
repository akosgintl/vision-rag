"""Pipeline router — orchestrates the full Retrieve -> Extract -> Generate flow."""

import uuid

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from proxy.config import settings
from proxy.models.requests import PipelineIngestRequest, PipelineQueryRequest
from proxy.models.responses import PipelineIngestResponse, PipelineQueryResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/v1/pipeline", tags=["pipeline"])


@router.post("/query", response_model=PipelineQueryResponse)
async def pipeline_query(req: PipelineQueryRequest, request: Request):
    """
    Full document understanding pipeline.

    1. ColPali retrieves top-K relevant pages from Qdrant
    2. Qwen3-VL extracts structured data from each page
    3. Qwen2.5-7B generates a natural-language answer

    This is the primary endpoint for end-user queries.
    """
    orchestrator = request.app.state.orchestrator

    try:
        result = await orchestrator.query(
            query=req.query,
            collection=req.collection,
            top_k=req.top_k,
            include_extractions=req.include_extractions,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return result

    except Exception as e:
        logger.error("pipeline_query_failed", error=str(e), query=req.query[:100])
        raise HTTPException(500, f"Pipeline query failed: {e}") from e


@router.post("/ingest", response_model=PipelineIngestResponse)
async def pipeline_ingest_json(req: PipelineIngestRequest, request: Request):
    """
    Ingest a new PDF document (by URL).

    Dispatches an async Celery task that:
    1. Downloads the PDF from the provided URL
    2. Rasterizes pages at specified DPI
    3. Generates ColPali embeddings for each page
    4. Indexes embeddings in Qdrant

    Returns a job_id for status tracking via GET /v1/pipeline/job/{job_id}.
    """
    job_id = str(uuid.uuid4())
    job_tracker = request.app.state.job_tracker

    # Create job record in Redis
    await job_tracker.create_job(
        job_id=job_id,
        collection=req.collection,
        document_url=req.document_url,
    )

    # Dispatch to Celery worker
    from proxy.tasks import ingest_document

    # URL-based ingest: the Celery worker will download the PDF.
    # For now, we require document_url. If not set, return error.
    if not req.document_url:
        raise HTTPException(400, "document_url is required for async ingestion. Use /ingest/upload for file uploads.")

    # Celery task expects hex-encoded bytes. For URL-based ingest,
    # we pass an empty payload — the worker downloads from the URL.
    ingest_document.delay(
        job_id=job_id,
        pdf_bytes_hex="",
        collection=req.collection,
        dpi=req.dpi,
        metadata=req.metadata,
    )

    logger.info(
        "ingest_job_dispatched",
        job_id=job_id,
        url=req.document_url,
        collection=req.collection,
    )

    return PipelineIngestResponse(
        job_id=job_id,
        status="pending",
        message=f"Ingestion job dispatched. Track status at /v1/pipeline/job/{job_id}",
    )


@router.post("/ingest/upload", response_model=PipelineIngestResponse)
async def pipeline_ingest_upload(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
    collection: str = Form(default="default"),
    dpi: int = Form(default=300),
):
    """
    Ingest a new PDF document via file upload.

    Accepts multipart/form-data with a PDF file.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file.size and file.size > max_bytes:
        raise HTTPException(400, f"File too large. Max size: {settings.max_upload_size_mb} MB")

    job_id = str(uuid.uuid4())
    pdf_bytes = await file.read()

    # Enforce size limit on actual bytes read (file.size may not be set by all clients)
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(400, f"File too large. Max size: {settings.max_upload_size_mb} MB")

    # Validate PDF magic bytes (%PDF header)
    if not pdf_bytes[:5].startswith(b"%PDF-"):
        raise HTTPException(400, "Invalid file: not a valid PDF (bad magic bytes).")

    job_tracker = request.app.state.job_tracker
    await job_tracker.create_job(
        job_id=job_id,
        collection=collection,
        filename=file.filename,
    )

    # For small files, process synchronously. For large files, dispatch to Celery.
    if len(pdf_bytes) > 10 * 1024 * 1024:  # > 10 MB → async
        from proxy.tasks import ingest_document

        ingest_document.delay(
            job_id=job_id,
            pdf_bytes_hex=pdf_bytes.hex(),
            collection=collection,
            dpi=dpi,
        )
        return PipelineIngestResponse(
            job_id=job_id,
            status="pending",
            message=f"Large file queued for async ingestion. Track at /v1/pipeline/job/{job_id}",
        )

    # Small file → process inline
    ingestion = request.app.state.ingestion_service
    try:
        result = await ingestion.ingest_pdf(
            pdf_bytes=pdf_bytes,
            collection=collection,
            document_id=job_id,
            dpi=dpi,
            job_id=job_id,
        )
        return PipelineIngestResponse(
            job_id=job_id,
            status=result["status"],
            message=f"Ingested {result['indexed_pages']}/{result['total_pages']} pages in {result['elapsed_seconds']}s",
        )
    except Exception as e:
        logger.error("ingest_upload_failed", error=str(e))
        await job_tracker.fail_job(job_id, str(e))
        raise HTTPException(500, f"Ingestion failed: {e}") from e


@router.get("/job/{job_id}")
async def pipeline_job_status(job_id: str, request: Request):
    """Check the status of an ingestion job."""
    job_tracker = request.app.state.job_tracker
    status = await job_tracker.get_status(job_id)

    if status is None:
        raise HTTPException(404, f"Job {job_id} not found.")

    return status


@router.get("/collections")
async def list_collections(request: Request):
    """List all document collections and their page counts."""
    index = request.app.state.embedding_index
    try:
        total = await index.count()
        return {
            "total_pages": total,
            "message": "Use collection-specific count endpoint for per-collection stats.",
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list collections: {e}") from e


@router.get("/collections/{collection_name}/count")
async def collection_count(collection_name: str, request: Request):
    """Get the number of indexed pages in a collection."""
    index = request.app.state.embedding_index
    try:
        count = await index.count(collection_name=collection_name)
        return {"collection": collection_name, "page_count": count}
    except Exception as e:
        raise HTTPException(500, f"Failed to count collection: {e}") from e


@router.get("/documents")
async def list_documents(
    request: Request,
    collection: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """List ingested documents with metadata from PostgreSQL."""
    metadata_db = request.app.state.metadata_db
    if not metadata_db:
        raise HTTPException(503, "Document metadata service not available.")

    try:
        docs = await metadata_db.list_documents(collection=collection, limit=limit, offset=offset)
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(500, f"Failed to list documents: {e}") from e


@router.get("/documents/{document_id}")
async def get_document(document_id: str, request: Request):
    """Get metadata for a specific document."""
    metadata_db = request.app.state.metadata_db
    if not metadata_db:
        raise HTTPException(503, "Document metadata service not available.")

    doc = await metadata_db.get_document(document_id)
    if not doc:
        raise HTTPException(404, f"Document {document_id} not found.")
    return doc
