"""
Configuration for the Vision-RAG Multi-Model Proxy.
All settings can be overridden via environment variables.
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelBackend(BaseModel):
    name: str
    url: str
    model_id: str
    port: int
    health_path: str = "/health"
    timeout: float = 120.0


class Settings(BaseSettings):
    """Application settings — override via env vars or .env file."""

    # ─── App ──────────────────────────────────────────
    app_name: str = "Vision-RAG Multi-Model Proxy"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # ─── CORS ─────────────────────────────────────────
    # Comma-separated origins, e.g. "https://app.example.com,https://admin.example.com"
    # Empty list = no CORS allowed (safe default). Set to ["*"] only for local dev.
    cors_origins: list[str] = []

    # ─── Model backends ──────────────────────────────
    colpali_url: str = "http://localhost:8001"
    colpali_model_id: str = "TomoroAI/tomoro-ai-colqwen3-embed-8b-awq"

    qwen3vl_url: str = "http://localhost:8002"
    qwen3vl_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"

    qwen25_url: str = "http://localhost:8003"
    qwen25_model_id: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"

    # ─── Timeouts ────────────────────────────────────
    backend_timeout: float = 120.0
    health_check_timeout: float = 5.0
    health_check_interval: int = 30

    # ─── Rate limiting ───────────────────────────────
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # ─── Pipeline defaults ───────────────────────────
    default_top_k: int = 5
    max_pages_per_request: int = 20
    max_concurrent_extractions: int = 5  # Bounded GPU parallelism
    max_image_size_mb: int = 10
    max_upload_size_mb: int = 100  # Max PDF upload size
    default_dpi: int = 300

    # ─── Infrastructure ──────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "doc_pages"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    postgres_dsn: str = "postgresql://visionrag:visionrag_secret@localhost:5432/visionrag"

    # ─── Circuit breaker ─────────────────────────────
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 30
    circuit_half_open_requests: int = 2

    # ─── API auth ────────────────────────────────────
    api_key: str | None = None  # Set to enable API key auth

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def backends(self) -> dict[str, ModelBackend]:
        return {
            "retrieve": ModelBackend(
                name="colpali",
                url=self.colpali_url,
                model_id=self.colpali_model_id,
                port=8001,
            ),
            "extract": ModelBackend(
                name="qwen3vl",
                url=self.qwen3vl_url,
                model_id=self.qwen3vl_model_id,
                port=8002,
            ),
            "generate": ModelBackend(
                name="qwen25",
                url=self.qwen25_url,
                model_id=self.qwen25_model_id,
                port=8003,
            ),
        }


settings = Settings()
