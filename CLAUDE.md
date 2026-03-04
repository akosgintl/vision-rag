# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visual-RAG is a GPU-optimized document understanding pipeline combining three vision-language models (ColPali, Qwen3-VL, Qwen2.5-7B) on a single NVIDIA A100 80GB GPU via vLLM, fronted by a FastAPI reverse proxy. It implements a four-stage pipeline: PDF rasterization → ColPali retrieval → Qwen3-VL extraction → Qwen2.5-7B generation.

## Development Commands

```bash
# Start full stack (requires NVIDIA GPU + Docker)
cp vision-rag/.env.example vision-rag/.env   # add HF_TOKEN
docker compose -f vision-rag/docker-compose.yml up -d

# Run proxy locally (with vLLM backends already running)
cd vision-rag && uvicorn proxy.main:app --host 0.0.0.0 --port 8000 --reload

# Start vLLM model servers manually
cd vision-rag && ./scripts/start_models.sh

# Health check
curl http://localhost:8000/health

# Run tests
cd vision-rag && pip install ".[test]" && pytest

# Run tests with coverage
cd vision-rag && pytest --cov --cov-report=term-missing

# Run a single test file
cd vision-rag && pytest tests/test_circuit_breaker.py -v

# Run a single test
cd vision-rag && pytest tests/test_circuit_breaker.py::TestCircuitBreaker::test_opens_after_threshold -v

# Lint
cd vision-rag && ruff check proxy/ tests/

# Auto-fix lint issues
cd vision-rag && ruff check --fix proxy/ tests/

# Format
cd vision-rag && ruff format proxy/ tests/
```

## Architecture

### Request Flow

```
Client → FastAPI Proxy (:8000)
           ├─→ ColPali vLLM (:8001)   — retrieval embeddings (multi-vector, 320-dim)
           ├─→ Qwen3-VL vLLM (:8002)  — page extraction (vision-language)
           └─→ Qwen2.5-7B vLLM (:8003) — text generation (AWQ 4-bit)
         + Qdrant (:6333)  — vector storage (MaxSim late-interaction scoring)
         + PostgreSQL       — metadata
         + MinIO            — page image storage (S3-compatible)
         + Redis            — task queue
```

### Code Layout (`vision-rag/proxy/`)

- **main.py** — FastAPI app with async lifespan that initializes all services, registers routers, and exposes health/metrics endpoints. Services are stored on `request.app.state`.
- **config.py** — `Settings` class using pydantic-settings; all values configurable via environment variables.
- **routers/** — Four routers mapping to the pipeline stages:
  - `retrieve.py` → `/v1/retrieve/*` (ColPali search & indexing via Qdrant)
  - `extract.py` → `/v1/extract/*` (Qwen3-VL structured extraction)
  - `generate.py` → `/v1/generate/*` (Qwen2.5 chat/summarize with SSE streaming)
  - `pipeline.py` → `/v1/pipeline/*` (orchestrated end-to-end query + PDF ingestion)
- **services/** — Core business logic:
  - `orchestrator.py` — Multi-stage pipeline: encode → search → fetch images → extract (concurrent `asyncio.gather`) → generate
  - `embedding_index.py` — Qdrant client wrapper for ColPali multi-vector storage with MaxSim comparator
  - `ingestion.py` — PDF rasterization (`pdf2image`) → embedding → Qdrant indexing
  - `circuit_breaker.py` — Per-backend resilience (CLOSED → OPEN → HALF_OPEN state machine)
  - `health.py` — Concurrent async health checks for all backends
- **middleware/** — `auth.py` (API key via hmac.compare_digest), `rate_limiter.py` (Redis sliding window), `request_id.py` (X-Request-ID propagation)
- **models/** — Pydantic request/response schemas in `requests.py` and `responses.py`

### Key Patterns

- **Dependency injection via app.state**: Services are created during lifespan startup and accessed through `request.app.state` in route handlers.
- **Async-first**: All backend calls use `httpx.AsyncClient` with connection pooling; extraction runs concurrently via `asyncio.gather`.
- **Middleware chain order**: RequestId (outermost) → CORS → Auth → Rate Limit → Prometheus instrumentation.
- **vLLM OpenAI compatibility**: All model backends expose `/v1/chat/completions` and `/v1/embeddings` endpoints.
- **Circuit breaker per backend**: Tracks failures per model server; opens after 5 failures, half-open recovery after 30s.

### GPU Memory Budget (A100 80GB)

ColPali ~8GB (FP16) + Qwen3-VL ~6GB (FP16) + Qwen2.5-7B ~14GB (AWQ 4-bit) = ~28GB for models, ~52GB available for KV-cache (with LMCache offloading to CPU RAM via `lmcache_config.yaml`).

### Infrastructure (Docker Compose)

The `docker-compose.yml` defines: 3 vLLM containers (GPU-shared), FastAPI proxy, Qdrant, PostgreSQL, MinIO, Redis, Prometheus, and Grafana. Health-check dependencies ensure the proxy only starts after all model servers are ready.

## Configuration

All settings are environment-configurable (see `vision-rag/.env.example`). Key variables:

- `HF_TOKEN` (required) — HuggingFace token for model downloads
- `API_KEY` (optional) — enables auth middleware when set
- `COLPALI_URL`, `QWEN3VL_URL`, `QWEN25_URL` — model backend URLs (auto-discovered in Docker)
- `BACKEND_TIMEOUT` (120s), `DEFAULT_DPI` (300), `DEFAULT_TOP_K` (5)
- `QDRANT_URL`, `REDIS_URL`, `MINIO_ENDPOINT`, `POSTGRES_DSN` — infrastructure endpoints

## Testing

Tests are in `vision-rag/tests/`. The test suite uses:
- **pytest** with `asyncio_mode = "auto"` (all async tests run automatically)
- **respx** for mocking `httpx.AsyncClient` calls
- **fakeredis** for Redis mocking (job tracker tests)
- **conftest.py** provides shared fixtures: `mock_backend`, `mock_embedding_index`, `mock_storage`, `mock_job_tracker`, `mock_metadata_db`, `client` (FastAPI TestClient with mocked services)

Coverage threshold: 65%. Currently ~76%. `proxy/main.py` and `proxy/tasks.py` are excluded from coverage (entrypoint and Celery worker).

## Experimentation Scripts (`scripts/`)

Standalone Jupyter notebooks for running individual pipeline stages interactively on RunPod GPU pods. These mirror the production model backends in `vision-rag/docker-compose.yml` but are designed for experimentation, benchmarking, and generating test artifacts.

### Layout

```
scripts/
├── colpali/                        # ColPali multi-vector embedding notebook
│   ├── README.md                   # Full documentation with architecture diagrams
│   ├── colqwen3_colpali_vllm.ipynb # Interactive embedding + MaxSim retrieval
│   └── embedding_output/           # Pre-computed outputs (meta JSON + page PNGs)
├── extract/                        # Qwen3-VL document extraction notebook
│   ├── README.md                   # Full documentation with architecture diagrams
│   ├── qwen3_vl_extract.ipynb      # Interactive content + figure extraction
│   └── extraction_output/          # Pre-computed outputs (extraction JSON + page PNGs + figure crops)
└── test/                           # Shared test documents
    ├── test_docx.docx              # 220-page Hungarian requirements doc (8 MB)
    ├── test_pdf.pdf                # 81-page PDF (842 KB)
    └── test_png.png                # Single PNG image (91 KB)
```

### ColPali Notebook (`scripts/colpali/`)

Generates multi-vector patch embeddings using `TomoroAI/tomoro-colqwen3-embed-8b` via vLLM (HTTP server or offline mode). Supports JPG, PNG, PDF, and DOCX inputs. Implements MaxSim scoring for retrieval ranking. Outputs per-page embedding metadata JSON and rasterized page images.

### Extraction Notebook (`scripts/extract/`)

Runs structured extraction using `Qwen/Qwen3-VL-8B-Instruct` via vLLM HTTP server. Two prompts per page: content extraction (Markdown + HTML tables) and figure extraction (bounding boxes + descriptions + cropped images). Outputs extraction JSON and page/figure PNGs.

### Running the Notebooks

Both notebooks require a GPU pod (RunPod recommended) with:
- **System deps**: `poppler-utils`, `libreoffice`, `libgl1`, `libglib2.0-0`
- **Python deps**: `vllm>=0.16.0`, `httpx>=0.27.0`, `Pillow`, `pdf2image`, `numpy`, `tqdm`, `opik`
- **vLLM server**: ColPali on `:8001` (task=embed), Qwen3-VL on `:8002` (task=generate)

See each notebook's README for full Quick Start instructions.

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes/PRs to main affecting `vision-rag/`:
1. **lint** — `ruff check` + `ruff format --check`
2. **test** — `pytest --cov`
3. **docker-build** — validates `docker build` succeeds
