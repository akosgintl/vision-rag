"""Shared test fixtures for the vision-rag test suite."""

from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

# ── Patch settings before any proxy import ──────────────────────


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Override settings so tests never touch real infrastructure."""
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://fake-qdrant:6333")
    monkeypatch.setenv("MINIO_ENDPOINT", "fake-minio:9000")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://test:test@fake:5432/test")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("DEBUG", "false")


# ── Fake Redis ──────────────────────────────────────────────────


@pytest.fixture
def fake_redis():
    """Provides a fakeredis async client."""
    server = fakeredis.FakeServer()
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


# ── HTTPX / respx mocking ──────────────────────────────────────


@pytest.fixture
def mock_httpx_client():
    """Provides a respx-mocked httpx.AsyncClient."""
    with respx.mock(assert_all_mocked=False) as mock:
        client = httpx.AsyncClient()
        yield client, mock


# ── Mock services ───────────────────────────────────────────────


@pytest.fixture
def mock_backend():
    """Mock BackendCaller."""
    backend = AsyncMock()
    backend.post = AsyncMock(
        return_value={
            "data": [{"embedding": [[0.1] * 128]}],
            "usage": {"total_tokens": 10},
            "choices": [{"message": {"content": "test response"}, "finish_reason": "stop"}],
        }
    )
    return backend


@pytest.fixture
def mock_embedding_index():
    """Mock EmbeddingIndex."""
    index = AsyncMock()
    index.ensure_collection = AsyncMock()
    index.search = AsyncMock(return_value=[])
    index.index_page = AsyncMock(return_value="test-point-id")
    index.count = AsyncMock(return_value=0)
    index.delete_document = AsyncMock()
    index.close = AsyncMock()
    return index


@pytest.fixture
def mock_storage():
    """Mock MinioStorage."""
    storage = AsyncMock()
    storage.ensure_bucket = MagicMock()
    storage.store_page_image = AsyncMock(return_value="default/doc1/page_1.png")
    storage.store_page_images = AsyncMock(return_value=3)
    storage.fetch_page_image = AsyncMock(return_value="base64encodedimage")
    return storage


@pytest.fixture
def mock_job_tracker():
    """Mock JobTracker."""
    tracker = AsyncMock()
    tracker.ping = AsyncMock(return_value=True)
    tracker.create_job = AsyncMock(return_value={"job_id": "test-job", "status": "pending"})
    tracker.get_status = AsyncMock(
        return_value={
            "job_id": "test-job",
            "status": "completed",
            "progress": 1.0,
            "total_pages": 3,
            "processed_pages": 3,
            "error": None,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:01+00:00",
        }
    )
    tracker.close = AsyncMock()
    return tracker


@pytest.fixture
def mock_metadata_db():
    """Mock DocumentMetadata."""
    db = AsyncMock()
    db.ensure_tables = MagicMock()
    db.register_document = AsyncMock(return_value={"document_id": "test-doc"})
    db.get_document = AsyncMock(return_value={"document_id": "test-doc", "collection": "default"})
    db.list_documents = AsyncMock(return_value=[{"document_id": "test-doc"}])
    db.close = MagicMock()
    return db


@pytest.fixture
def mock_health_service():
    """Mock HealthService."""
    svc = AsyncMock()
    svc.check_all = AsyncMock()
    svc.check_infra = AsyncMock(return_value={})
    svc.readiness = AsyncMock(return_value={"ready": True, "status": "ready"})
    return svc


# ── FastAPI test app ────────────────────────────────────────────


@pytest.fixture
def app_with_mocks(
    mock_backend,
    mock_embedding_index,
    mock_storage,
    mock_job_tracker,
    mock_metadata_db,
    mock_health_service,
):
    """
    Create a FastAPI test app with all services mocked out.

    Patches lifespan so we skip real infra connections entirely.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from proxy.routers import extract, generate, pipeline, retrieve
    from proxy.services.circuit_breaker import CircuitBreakerRegistry

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient()
        app.state.backend = mock_backend
        app.state.embedding_index = mock_embedding_index
        app.state.storage = mock_storage
        app.state.job_tracker = mock_job_tracker
        app.state.metadata_db = mock_metadata_db
        app.state.health_service = mock_health_service
        app.state.circuit_breakers = CircuitBreakerRegistry()
        app.state.orchestrator = AsyncMock()
        app.state.ingestion_service = AsyncMock()
        yield
        await app.state.client.aclose()

    app = FastAPI(lifespan=test_lifespan)
    app.include_router(retrieve.router)
    app.include_router(extract.router)
    app.include_router(generate.router)
    app.include_router(pipeline.router)

    # Add simple health endpoints used by tests
    @app.get("/healthz")
    async def liveness():
        return {"status": "alive"}

    @app.get("/health")
    async def health(request):
        return await request.app.state.health_service.check_all()

    return app


@pytest.fixture
def client(app_with_mocks):
    """FastAPI TestClient with mocked services."""
    with TestClient(app_with_mocks) as c:
        yield c
