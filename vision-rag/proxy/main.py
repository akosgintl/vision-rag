"""
Vision-RAG Multi-Model Proxy — Main Application

Serves as the unified gateway for three vLLM model backends:
  - ColPali (TomoroAI/tomoro-ai-colqwen3-embed-8b-awq)  → Visual retrieval
  - Qwen3-VL-2B-Instruct                                → Visual extraction
  - Qwen2.5-7B-Instruct-AWQ                             → Text generation

All three share a single NVIDIA A100 80GB GPU.
"""

from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import ValidationError

from proxy.config import settings
from proxy.middleware.auth import AuthMiddleware
from proxy.middleware.rate_limiter import RateLimiterMiddleware
from proxy.middleware.request_id import RequestIdMiddleware
from proxy.routers import extract, generate, pipeline, retrieve
from proxy.services.backend import BackendCaller, BackendError
from proxy.services.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from proxy.services.embedding_index import EmbeddingIndex
from proxy.services.health import HealthService, wait_for_backends
from proxy.services.ingestion import IngestionService
from proxy.services.job_tracker import JobTracker
from proxy.services.metadata import DocumentMetadata
from proxy.services.orchestrator import PipelineOrchestrator
from proxy.services.storage import MinioStorage

# ─── Structured logging ──────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()


# ─── Application lifespan ────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("starting_proxy", version=settings.app_version)

    # HTTP client for backend communication
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    # Wait for all vLLM backends to be healthy
    try:
        await wait_for_backends(app.state.client, max_retries=60, interval=5.0)
    except RuntimeError as e:
        logger.error("backend_startup_failed", error=str(e))
        # Continue anyway — some backends might come up later

    # Initialize services
    app.state.health_service = HealthService(app.state.client)
    app.state.circuit_breakers = CircuitBreakerRegistry(
        failure_threshold=settings.circuit_failure_threshold,
        recovery_timeout=settings.circuit_recovery_timeout,
    )

    # Resilient backend caller (circuit breaker + retry)
    app.state.backend = BackendCaller(
        client=app.state.client,
        circuit_breakers=app.state.circuit_breakers,
    )

    # Embedding index (Qdrant — async)
    app.state.embedding_index = EmbeddingIndex()
    try:
        await app.state.embedding_index.ensure_collection()
    except Exception as e:
        logger.warning("qdrant_init_failed", error=str(e))

    # MinIO object storage
    app.state.storage = MinioStorage()
    try:
        app.state.storage.ensure_bucket()
    except Exception as e:
        logger.warning("minio_init_failed", error=str(e))

    # Redis job tracker
    app.state.job_tracker = JobTracker()
    try:
        if await app.state.job_tracker.ping():
            logger.info("redis_connected")
        else:
            logger.warning("redis_ping_failed")
    except Exception as e:
        logger.warning("redis_init_failed", error=str(e))

    # PostgreSQL document metadata
    app.state.metadata_db = None
    try:
        app.state.metadata_db = DocumentMetadata()
        app.state.metadata_db.ensure_tables()
    except Exception as e:
        logger.warning("postgres_init_failed", error=str(e))

    # Pipeline orchestrator
    app.state.orchestrator = PipelineOrchestrator(
        backend=app.state.backend,
        embedding_index=app.state.embedding_index,
        storage=app.state.storage,
    )

    # Ingestion service
    app.state.ingestion_service = IngestionService(
        backend=app.state.backend,
        embedding_index=app.state.embedding_index,
        storage=app.state.storage,
        job_tracker=app.state.job_tracker,
        metadata_db=app.state.metadata_db,
    )

    logger.info("proxy_ready", port=settings.port)
    yield

    # Shutdown
    await app.state.client.aclose()
    await app.state.embedding_index.close()
    await app.state.job_tracker.close()
    if app.state.metadata_db:
        app.state.metadata_db.close()
    logger.info("proxy_shutdown")


# ─── FastAPI app ─────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Unified proxy for multi-model document understanding pipeline. "
        "Routes requests to ColPali, Qwen3-VL, and Qwen2.5-7B via vLLM."
    ),
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimiterMiddleware)
# RequestId outermost (added last = runs first) so all middleware/handlers have it
app.add_middleware(RequestIdMiddleware)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)


# ─── Structured error handlers ──────────────────────────────


def _get_request_id(request) -> str:
    return getattr(request.state, "request_id", "unknown") if hasattr(request, "state") else "unknown"


@app.exception_handler(CircuitOpenError)
async def circuit_open_handler(request, exc: CircuitOpenError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "code": "CIRCUIT_OPEN",
            "detail": f"Backend '{exc.backend}' is temporarily unavailable.",
            "retry_after": round(exc.retry_after),
            "request_id": _get_request_id(request),
        },
        headers={"Retry-After": str(round(exc.retry_after))},
    )


@app.exception_handler(BackendError)
async def backend_error_handler(request, exc: BackendError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code or 502,
        content={
            "error": "Bad Gateway",
            "code": "BACKEND_ERROR",
            "detail": exc.message if settings.debug else f"Backend '{exc.backend}' request failed.",
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "code": "VALIDATION_ERROR",
            "detail": exc.errors() if settings.debug else "Request validation failed.",
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    from fastapi.responses import JSONResponse

    logger.error("unhandled_exception", error=str(exc), exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "code": "INTERNAL_ERROR",
            "detail": str(exc) if settings.debug else "An unexpected error occurred.",
            "request_id": _get_request_id(request),
        },
    )


# ─── Routers ─────────────────────────────────────────────────

app.include_router(retrieve.router)
app.include_router(extract.router)
app.include_router(generate.router)
app.include_router(pipeline.router)


# ─── Root endpoints ──────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "retrieve": "/v1/retrieve/search",
            "extract": "/v1/extract/page",
            "generate": "/v1/generate/chat",
            "pipeline": "/v1/pipeline/query",
            "health": "/health",
            "healthz": "/healthz",
            "readyz": "/readyz",
            "docs": "/docs",
        },
    }


@app.get("/healthz")
async def liveness():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "alive"}


@app.get("/readyz")
async def readiness(request):
    """
    Readiness probe — checks all backends + infrastructure.

    Returns 200 if ready, 503 if not ready.
    """
    from fastapi.responses import JSONResponse

    result = await request.app.state.health_service.readiness()
    status_code = 200 if result["ready"] else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/health")
async def health(request):
    """Health check for all model backends."""
    return await request.app.state.health_service.check_all()


@app.get("/health/infra")
async def infra_health(request):
    """Health check for infrastructure dependencies (Qdrant, Redis, MinIO, PostgreSQL)."""
    return await request.app.state.health_service.check_infra()


@app.get("/health/circuits")
async def circuit_status(request):
    """Circuit breaker status for all backends."""
    return request.app.state.circuit_breakers.status()


@app.get("/health/{backend}")
async def health_backend(backend: str, request):
    """Health check for a specific model backend."""
    if backend not in settings.backends:
        from fastapi import HTTPException

        raise HTTPException(404, f"Unknown backend: {backend}. Valid: {list(settings.backends.keys())}")
    return await request.app.state.health_service.check_backend(backend)


# ─── Direct proxy passthrough ────────────────────────────────


@app.api_route("/v1/proxy/{service}/{path:path}", methods=["GET", "POST"])
async def raw_proxy(service: str, path: str, request):
    """
    Raw passthrough proxy to vLLM backends.

    Use this to access the native vLLM OpenAI-compatible API directly.
    Example: POST /v1/proxy/generate/chat/completions
    """
    if service not in settings.backends:
        from fastapi import HTTPException

        raise HTTPException(404, f"Unknown service: {service}")

    backend = settings.backends[service]
    backend_url = f"{backend.url}/v1/{path}"
    body = await request.body()

    response = await request.app.state.client.request(
        method=request.method,
        url=backend_url,
        content=body,
        headers={"content-type": request.headers.get("content-type", "application/json")},
    )

    from fastapi.responses import Response

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )


# ─── Run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "proxy.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
