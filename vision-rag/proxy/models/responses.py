"""Response schemas for all API endpoints."""

from datetime import datetime

from pydantic import BaseModel


class SourceReference(BaseModel):
    document_id: str
    page_number: int
    relevance_score: float
    collection: str


class TokenUsage(BaseModel):
    retrieval_tokens: int = 0
    extraction_tokens: int = 0
    generation_tokens: int = 0

    @property
    def total(self) -> int:
        return self.retrieval_tokens + self.extraction_tokens + self.generation_tokens


# ─── Retrieval Responses ─────────────────────────────────────


class RetrievalResult(BaseModel):
    document_id: str
    page_number: int
    score: float
    collection: str


class RetrievalSearchResponse(BaseModel):
    results: list[RetrievalResult]
    query: str
    latency_ms: float


class RetrievalIndexResponse(BaseModel):
    document_id: str
    page_number: int
    status: str = "indexed"
    embedding_dim: int | None = None


# ─── Extraction Responses ────────────────────────────────────


class ExtractionResult(BaseModel):
    content: str
    format: str
    tokens_used: int
    latency_ms: float


class ExtractBatchResponse(BaseModel):
    results: list[ExtractionResult]
    total_tokens: int
    total_latency_ms: float


# ─── Generation Responses ────────────────────────────────────


class GenerateChatResponse(BaseModel):
    content: str
    tokens_used: int
    finish_reason: str
    latency_ms: float


class GenerateSummarizeResponse(BaseModel):
    summary: str
    style: str
    tokens_used: int
    latency_ms: float


# ─── Pipeline Responses ──────────────────────────────────────


class PipelineQueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    extractions: list[str] | None = None
    latency_ms: float
    tokens: TokenUsage


class PipelineJobStatus(BaseModel):
    job_id: str
    status: str  # pending | processing | completed | failed
    progress: float | None = None
    total_pages: int | None = None
    processed_pages: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class PipelineIngestResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Document ingestion job created."


# ─── Health ──────────────────────────────────────────────────


class BackendStatus(BaseModel):
    name: str
    status: str  # healthy | unhealthy | unreachable
    latency_ms: float | None = None
    model_id: str | None = None


class HealthResponse(BaseModel):
    status: str  # ok | degraded | down
    backends: list[BackendStatus]
    uptime_seconds: float
    version: str
