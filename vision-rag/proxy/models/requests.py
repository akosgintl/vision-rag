"""Request schemas for all API endpoints."""

from pydantic import BaseModel, Field, field_validator

from proxy.prompts import ExtractionPrompts

# Max base64 size: ~20MB image → ~27MB base64 encoded
MAX_BASE64_LENGTH = 28_000_000
MAX_PROMPT_LENGTH = 10_000
MAX_QUERY_LENGTH = 5_000
MAX_CONTENT_LENGTH = 500_000  # ~500KB of text
MAX_BATCH_SIZE = 50
MAX_MESSAGES = 100
MAX_DOCUMENT_ID_LENGTH = 256


# ─── Retrieval (ColPali) ─────────────────────────────────────


class RetrievalSearchRequest(BaseModel):
    """Search for relevant document pages using a text query."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="Natural language search query")
    collection: str = Field(
        default="default", min_length=1, max_length=128, description="Document collection to search"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class RetrievalIndexRequest(BaseModel):
    """Index new document pages into the embedding store."""

    document_id: str = Field(..., min_length=1, max_length=MAX_DOCUMENT_ID_LENGTH)
    page_number: int = Field(..., ge=0, le=10_000)
    image_base64: str = Field(..., max_length=MAX_BASE64_LENGTH, description="Base64-encoded page image (PNG/JPEG)")
    collection: str = Field(default="default", min_length=1, max_length=128)
    metadata: dict | None = None


class RetrievalBatchIndexRequest(BaseModel):
    """Index multiple pages at once."""

    pages: list[RetrievalIndexRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


# ─── Extraction (Qwen3-VL) ──────────────────────────────────


class ExtractPageRequest(BaseModel):
    """Extract structured data from a single document page."""

    image_base64: str = Field(..., max_length=MAX_BASE64_LENGTH, description="Base64-encoded page image")
    prompt: str | None = Field(
        default=ExtractionPrompts.DEFAULT,
        max_length=MAX_PROMPT_LENGTH,
        description="Custom extraction prompt",
    )
    output_format: str = Field(default="json", description="json | markdown | text")
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        allowed = {"json", "markdown", "text"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return v


class ExtractBatchRequest(BaseModel):
    """Extract from multiple pages."""

    pages: list[ExtractPageRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    concurrent: bool = Field(default=True, description="Process pages concurrently")


# ─── Generation (Qwen2.5-7B) ────────────────────────────────


class GenerateChatRequest(BaseModel):
    """Chat completion with context."""

    messages: list[dict] = Field(
        ..., min_length=1, max_length=MAX_MESSAGES, description="OpenAI-compatible messages array"
    )
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    stream: bool = Field(default=False)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[dict]) -> list[dict]:
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if msg["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid role: {msg['role']}. Must be system, user, or assistant")
            if isinstance(msg["content"], str) and len(msg["content"]) > MAX_CONTENT_LENGTH:
                raise ValueError(f"Message content exceeds max length of {MAX_CONTENT_LENGTH}")
        return v


class GenerateSummarizeRequest(BaseModel):
    """Summarize extracted document content."""

    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH, description="Extracted content to summarize")
    style: str = Field(default="concise", description="concise | detailed | bullet_points")
    max_tokens: int = Field(default=512, ge=64, le=4096)

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        allowed = {"concise", "detailed", "bullet_points"}
        if v not in allowed:
            raise ValueError(f"style must be one of {allowed}")
        return v


# ─── Pipeline (Full Orchestration) ──────────────────────────


class PipelineQueryRequest(BaseModel):
    """Full pipeline: Retrieve → Extract → Generate."""

    query: str = Field(
        ..., min_length=1, max_length=MAX_QUERY_LENGTH, description="Natural language question about documents"
    )
    collection: str = Field(
        default="default", min_length=1, max_length=128, description="Document collection to search"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Pages to retrieve")
    include_extractions: bool = Field(default=True, description="Return raw extractions")
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class PipelineIngestRequest(BaseModel):
    """Ingest a new PDF document (async)."""

    document_url: str | None = Field(None, max_length=2048, description="S3/MinIO URL of the PDF")
    collection: str = Field(default="default", min_length=1, max_length=128)
    metadata: dict | None = None
    dpi: int = Field(default=300, ge=72, le=600, description="Rasterization DPI")
