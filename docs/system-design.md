# Multi-Model vLLM Deployment Architecture
## Document Understanding at Scale — HLD & LLD

---

## Executive Summary

This document describes the architecture for a **GPU-shared, multi-model document understanding pipeline** that combines three specialized models behind a unified FastAPI reverse proxy, all served via vLLM on a single NVIDIA A100 80GB GPU.

| Role | Model | VRAM (AWQ 4-bit) | Port |
|---|---|---|---|
| Retrieval | ColPali (TomoroAI/tomoro-ai-colqwen3-embed-8b-awq) | ~8 GB | 8001 |
| Visual Extraction | Qwen3-VL-2B-Instruct | ~6 GB | 8002 |
| Text Generation / RAG | Qwen2.5-7B-Instruct | ~14 GB | 8003 |
| **Total** | | **~28 GB** | |

> **Why Qwen2.5-7B-Instruct as the third model?**
> - Strong reasoning and instruction-following for RAG answer synthesis
> - Excellent multilingual support (matches Qwen3-VL ecosystem)
> - Fits comfortably on remaining VRAM after ColPali + Qwen3-VL
> - Proven performance on summarization, Q&A, and structured output tasks
> - Native tool-calling support for agentic workflows

---

# PART 1 — HIGH-LEVEL DESIGN (HLD)

---

## 1.1 System Context Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENTS                                │
│   Web App  ·  API Consumers  ·  Batch Jobs  ·  CLI Tools     │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS / REST
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              NGINX / Traefik (TLS Termination)               │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           FastAPI Reverse Proxy Router (:8000)                │
│                                                               │
│   /v1/retrieve/*  ──→  ColPali vLLM        (:8001)           │
│   /v1/extract/*   ──→  Qwen3-VL vLLM      (:8002)           │
│   /v1/generate/*  ──→  Qwen2.5-7B vLLM    (:8003)           │
│   /v1/pipeline/*  ──→  Orchestrator (all 3)                  │
│                                                               │
│   Health checks · Rate limiting · Request queuing             │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ vLLM    │   │ vLLM    │   │ vLLM    │
   │ ColPali │   │ Qwen3-VL│   │Qwen2.5  │
   │ :8001   │   │ :8002   │   │ :8003   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
            ┌────────────────┐
            │  NVIDIA A100   │
            │    80 GB GPU   │
            │                │
            │  LMCache       │
            │  (Shared KV)   │
            └────────────────┘
```

## 1.2 Core Pipeline Flow

```
PDF Upload ──→ Page Rasterization ──→ ColPali Retrieval ──→ Qwen3-VL Extraction ──→ Qwen2.5 Generation ──→ Response
   │                 │                      │                       │                        │
   │            pdf2image              Visual embeddings       Key-Value pairs         Natural language
   │            (300 DPI)              + late-interaction       from doc pages          answer / summary
   ▼                                   scoring
 Object Store                                              
 (MinIO/S3)                                                
```

### Pipeline Stages

**Stage 1 — Ingestion:** Raw PDFs are uploaded, split into pages, and rasterized at 300 DPI. Page images are stored in object storage (MinIO/S3). Metadata is indexed in PostgreSQL.

**Stage 2 — Retrieval (ColPali):** Each page image is encoded into multi-vector embeddings by ColPali. At query time, the user's text query is also encoded, and late-interaction scoring identifies the most relevant pages. This replaces traditional OCR → chunking → vector search.

**Stage 3 — Extraction (Qwen3-VL):** The top-K retrieved page images are passed to Qwen3-VL with structured prompts. The VLM extracts key-value pairs, tables, figures, and text in a structured JSON format.

**Stage 4 — Generation (Qwen2.5-7B):** The extracted structured data is fed as context to Qwen2.5-7B-Instruct, which generates natural-language answers, summaries, comparisons, or fills templates.

## 1.3 Component Overview

| Component | Technology | Responsibility |
|---|---|---|
| Reverse Proxy | FastAPI + httpx | Route requests, load balance, health checks |
| Model Serving | vLLM (3 instances) | GPU inference with PagedAttention |
| Memory Optimization | LMCache | Shared KV-cache via system RAM |
| Document Storage | MinIO / S3 | Store original PDFs and rasterized pages |
| Metadata Store | PostgreSQL | Document metadata, job tracking |
| Embedding Index | Qdrant / FAISS | ColPali embedding storage and retrieval |
| Task Queue | Redis + Celery | Async pipeline orchestration |
| Monitoring | Prometheus + Grafana | GPU util, latency, throughput metrics |
| Containerization | Docker Compose / K8s | Service orchestration |

## 1.4 Non-Functional Requirements

| Requirement | Target |
|---|---|
| Latency (single query, retrieval) | < 200ms |
| Latency (full pipeline, per page) | < 3s |
| Throughput | 50 concurrent users |
| GPU Utilization | > 85% |
| Availability | 99.5% |
| Max document size | 500 pages |

---

# PART 2 — LOW-LEVEL DESIGN (LLD)

---

## 2.1 VRAM Budget & Model Configuration

### GPU Memory Layout (A100 80GB)

```
┌─────────────────────────────────────────────────────────────┐
│                    A100 80GB VRAM                            │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   ColPali    │   Qwen3-VL   │  Qwen2.5-7B  │   KV-Cache +  │
│ TomColQwen3  │   2B-Instruct│   AWQ 4-bit   │   Overhead    │
│              │              │               │               │
│   ~8 GB      │   ~6 GB      │   ~14 GB      │   ~52 GB      │
│  (FP16)      │  (FP16)      │  (AWQ)        │  (Dynamic)    │
├──────────────┴──────────────┴──────────────┴────────────────┤
│  Total Model Weights: ~28 GB  |  Available for KV: ~52 GB   │
└─────────────────────────────────────────────────────────────┘
```

### vLLM Launch Commands

```bash
# Model 1: ColPali (Retrieval)
vllm serve TomoroAI/tomoro-ai-colqwen3-embed-8b-awq \
  --port 8001 \
  --gpu-memory-utilization 0.10 \
  --max-model-len 8192 \
  --dtype float16 \
  --trust-remote-code \
  --task embed \
  --override-pooler-config '{"pooling_type":"ALL"}' \
  --disable-log-requests

# Model 2: Qwen3-VL (Visual Extraction)
vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.08 \
  --max-model-len 4096 \
  --dtype float16 \
  --limit-mm-per-prompt image=5 \
  --trust-remote-code \
  --disable-log-requests

# Model 3: Qwen2.5-7B (Text Generation)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8003 \
  --gpu-memory-utilization 0.18 \
  --max-model-len 8192 \
  --quantization awq \
  --trust-remote-code \
  --enable-chunked-prefill \
  --disable-log-requests
```

> **Note:** `gpu-memory-utilization` values are conservative starting points. The remaining ~64% is available for KV-cache across all three instances. LMCache offloads cold KV entries to system RAM.

### LMCache Configuration

```yaml
# lmcache_config.yaml
chunk_size: 256
local_device: "cpu"        # Offload to system RAM
remote_url: null            # Single-node, no remote
remote_serde: null
enable_prefix_caching: true

# Environment variables
LMC_CONFIG_FILE: /etc/lmcache/config.yaml
```

## 2.2 FastAPI Reverse Proxy — Detailed Design

### Project Structure

```
proxy/
├── main.py                 # FastAPI app entry point
├── config.py               # Model endpoints, timeouts
├── routers/
│   ├── retrieve.py         # /v1/retrieve/* → ColPali
│   ├── extract.py          # /v1/extract/*  → Qwen3-VL
│   ├── generate.py         # /v1/generate/* → Qwen2.5-7B
│   └── pipeline.py         # /v1/pipeline/* → Orchestrated
├── middleware/
│   ├── rate_limiter.py     # Token bucket rate limiting
│   ├── auth.py             # API key validation
│   └── logging.py          # Structured request logging
├── services/
│   ├── health.py           # Health check logic
│   ├── queue.py            # Request queuing
│   └── orchestrator.py     # Multi-model pipeline
├── models/
│   ├── requests.py         # Pydantic request schemas
│   └── responses.py        # Pydantic response schemas
└── tests/
```

### Core Proxy Logic (main.py)

```python
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
import httpx
import asyncio

MODEL_BACKENDS = {
    "retrieve": "http://localhost:8001",
    "extract":  "http://localhost:8002",
    "generate": "http://localhost:8003",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=120.0)
    # Wait for all backends to be healthy
    await wait_for_backends(app.state.client)
    yield
    await app.state.client.aclose()

app = FastAPI(title="Vision-RAG Proxy", lifespan=lifespan)

@app.api_route("/v1/{service}/{path:path}", methods=["GET", "POST"])
async def proxy(service: str, path: str, request: Request):
    if service not in MODEL_BACKENDS:
        raise HTTPException(404, f"Unknown service: {service}")
    
    backend_url = f"{MODEL_BACKENDS[service]}/v1/{path}"
    body = await request.body()
    
    response = await request.app.state.client.request(
        method=request.method,
        url=backend_url,
        content=body,
        headers={
            "content-type": request.headers.get("content-type", "application/json")
        },
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )

@app.get("/health")
async def health():
    statuses = {}
    for name, url in MODEL_BACKENDS.items():
        try:
            r = await app.state.client.get(f"{url}/health", timeout=5.0)
            statuses[name] = "healthy" if r.status_code == 200 else "unhealthy"
        except Exception:
            statuses[name] = "unreachable"
    
    all_healthy = all(s == "healthy" for s in statuses.values())
    return {"status": "ok" if all_healthy else "degraded", "backends": statuses}
```

### Pipeline Orchestrator (services/orchestrator.py)

```python
import base64
import httpx
from typing import List

class PipelineOrchestrator:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    async def run(self, query: str, page_images: List[str]) -> dict:
        """
        Full pipeline: Retrieve → Extract → Generate
        
        Args:
            query: User's natural language question
            page_images: List of base64-encoded page images
        """
        # Step 1: Retrieve top-K relevant pages via ColPali
        retrieval_results = await self._retrieve(query, page_images)
        top_pages = retrieval_results[:5]  # Top 5 pages
        
        # Step 2: Extract structured data from top pages via Qwen3-VL
        extractions = await asyncio.gather(*[
            self._extract(page["image"], query) for page in top_pages
        ])
        
        # Step 3: Generate answer via Qwen2.5-7B
        context = "\n\n".join([
            f"[Page {i+1}]\n{ext}" for i, ext in enumerate(extractions)
        ])
        answer = await self._generate(query, context)
        
        return {
            "answer": answer,
            "sources": [p["page_id"] for p in top_pages],
            "extractions": extractions,
        }
    
    async def _retrieve(self, query: str, images: List[str]) -> List[dict]:
        """Encode query + images via ColPali, compute similarity."""
        response = await self.client.post(
            "http://localhost:8001/v1/embeddings",
            json={
                "model": "TomoroAI/tomoro-ai-colqwen3-embed-8b-awq",
                "input": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in images
                ] + [{"type": "text", "text": query}],
            },
        )
        # Late-interaction scoring between query and page embeddings
        data = response.json()
        return self._score_and_rank(data, len(images))
    
    async def _extract(self, image_b64: str, query: str) -> str:
        """Extract structured info from a page via Qwen3-VL."""
        response = await self.client.post(
            "http://localhost:8002/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-VL-2B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            {"type": "text", "text": f"Extract all relevant information from this document page related to: {query}. Return as structured JSON with keys: tables, key_values, text_blocks."},
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
            },
        )
        return response.json()["choices"][0]["message"]["content"]
    
    async def _generate(self, query: str, context: str) -> str:
        """Generate final answer via Qwen2.5-7B."""
        response = await self.client.post(
            "http://localhost:8003/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [
                    {"role": "system", "content": "You are a document analysis assistant. Answer questions based solely on the provided document extractions. Cite page numbers."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
        )
        return response.json()["choices"][0]["message"]["content"]
```

## 2.3 Docker Compose Deployment

```yaml
# docker-compose.yaml
version: "3.9"

services:
  # ─── MODEL SERVERS ─────────────────────────────────
  colpali:
    image: vllm/vllm-openai:latest
    command: >
      --model TomoroAI/tomoro-ai-colqwen3-embed-8b-awq
      --port 8001
      --gpu-memory-utilization 0.10
      --max-model-len 8192
      --dtype float16
      --task embed
      --override-pooler-config '{"pooling_type":"ALL"}'
      --trust-remote-code
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface
      - ./lmcache_config.yaml:/etc/lmcache/config.yaml
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LMC_CONFIG_FILE=/etc/lmcache/config.yaml
    networks:
      - model-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  qwen3vl:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen3-VL-2B-Instruct
      --port 8002
      --gpu-memory-utilization 0.08
      --max-model-len 4096
      --dtype float16
      --limit-mm-per-prompt image=5
      --trust-remote-code
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface
      - ./lmcache_config.yaml:/etc/lmcache/config.yaml
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LMC_CONFIG_FILE=/etc/lmcache/config.yaml
    networks:
      - model-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  qwen25:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-7B-Instruct-AWQ
      --port 8003
      --gpu-memory-utilization 0.18
      --max-model-len 8192
      --quantization awq
      --trust-remote-code
      --enable-chunked-prefill
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface
      - ./lmcache_config.yaml:/etc/lmcache/config.yaml
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LMC_CONFIG_FILE=/etc/lmcache/config.yaml
    networks:
      - model-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # ─── PROXY ─────────────────────────────────────────
  proxy:
    build: ./proxy
    ports:
      - "8000:8000"
    depends_on:
      colpali:
        condition: service_healthy
      qwen3vl:
        condition: service_healthy
      qwen25:
        condition: service_healthy
    environment:
      - COLPALI_URL=http://colpali:8001
      - QWEN3VL_URL=http://qwen3vl:8002
      - QWEN25_URL=http://qwen25:8003
    networks:
      - model-net

  # ─── INFRASTRUCTURE ────────────────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - model-net

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data
    networks:
      - model-net

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: visionrag
      POSTGRES_USER: visionrag
      POSTGRES_PASSWORD: visionrag_secret
    volumes:
      - pg-data:/var/lib/postgresql/data
    networks:
      - model-net

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage
    networks:
      - model-net

  # ─── MONITORING ────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - model-net

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    networks:
      - model-net

volumes:
  model-cache:
  minio-data:
  pg-data:
  qdrant-data:

networks:
  model-net:
    driver: bridge
```

## 2.4 API Schema Design

### Endpoints

```
POST /v1/retrieve/search          → ColPali similarity search
POST /v1/retrieve/index           → Index new document pages
POST /v1/extract/page             → Extract from single page
POST /v1/extract/batch            → Extract from multiple pages
POST /v1/generate/chat            → Chat completion
POST /v1/generate/summarize       → Summarize extracted content
POST /v1/pipeline/query           → Full pipeline (retrieve+extract+generate)
POST /v1/pipeline/ingest          → Ingest new PDF (async job)
GET  /v1/pipeline/job/{job_id}    → Check job status
GET  /health                      → Health check all backends
GET  /metrics                     → Prometheus metrics
```

### Request/Response Schemas

```python
# models/requests.py
from pydantic import BaseModel
from typing import Optional

class PipelineQueryRequest(BaseModel):
    query: str
    collection: str                    # Which document collection to search
    top_k: int = 5                     # Number of pages to retrieve
    include_extractions: bool = True   # Return raw extractions
    max_tokens: int = 1024             # Max generation tokens
    temperature: float = 0.3

class IngestRequest(BaseModel):
    document_url: str                  # S3/MinIO URL or upload
    collection: str                    # Target collection
    metadata: Optional[dict] = None

class ExtractPageRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = "Extract all structured data."
    output_format: str = "json"        # json | markdown | text

# models/responses.py
class PipelineQueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    extractions: Optional[list[str]]
    latency_ms: float
    tokens_used: TokenUsage

class SourceReference(BaseModel):
    document_id: str
    page_number: int
    relevance_score: float

class TokenUsage(BaseModel):
    retrieval_tokens: int
    extraction_tokens: int
    generation_tokens: int
    total: int
```

## 2.5 Embedding Index Design (Qdrant)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    CollectionDescription
)

client = QdrantClient("localhost", port=6333)

# ColPali produces multi-vector embeddings (128-dim per patch token)
# We store a flattened representation per page

client.create_collection(
    collection_name="doc_pages",
    vectors_config={
        "colpali": VectorParams(
            size=128,           # ColPali embedding dimension
            distance=Distance.COSINE,
            multivector_config={
                "comparator": "max_sim",  # Late-interaction scoring
            },
        ),
    },
)

# Indexing a page
def index_page(page_id: str, embeddings: list, metadata: dict):
    client.upsert(
        collection_name="doc_pages",
        points=[
            PointStruct(
                id=page_id,
                vector={"colpali": embeddings},  # Multi-vector
                payload={
                    "document_id": metadata["document_id"],
                    "page_number": metadata["page_number"],
                    "collection": metadata["collection"],
                    "ingested_at": metadata["timestamp"],
                },
            )
        ],
    )
```

## 2.6 Monitoring & Observability

### Key Metrics to Track

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm-colpali'
    static_configs:
      - targets: ['colpali:8001']
    metrics_path: /metrics

  - job_name: 'vllm-qwen3vl'
    static_configs:
      - targets: ['qwen3vl:8002']
    metrics_path: /metrics

  - job_name: 'vllm-qwen25'
    static_configs:
      - targets: ['qwen25:8003']
    metrics_path: /metrics

  - job_name: 'proxy'
    static_configs:
      - targets: ['proxy:8000']
    metrics_path: /metrics
```

### Critical Dashboards

| Dashboard | Panels |
|---|---|
| GPU Health | VRAM usage per model, GPU utilization %, temperature |
| Inference | P50/P95/P99 latency per model, tokens/sec, queue depth |
| Pipeline | End-to-end latency, success rate, error rate by stage |
| Business | Queries/min, documents ingested/day, unique users |

## 2.7 Error Handling & Resilience

```
┌────────────┐     Timeout?      ┌──────────┐
│  Request    │────────────────→  │  Retry   │──→ (max 2 retries)
│  arrives    │                   │  with    │
└──────┬─────┘                   │ backoff  │
       │                         └──────────┘
       ▼                              │
  Model healthy?                 Still failing?
       │                              │
  Yes ─┤── No ──→ Return 503         ▼
       │          "Model unavailable" Circuit breaker opens
       ▼                              │
  Process request                     ▼
       │                         Return cached result
       ▼                         or degraded response
  Return response
```

### Circuit Breaker Config

```python
CIRCUIT_BREAKER = {
    "failure_threshold": 5,      # Open after 5 consecutive failures
    "recovery_timeout": 30,      # Try again after 30s
    "half_open_requests": 2,     # Allow 2 test requests
}
```

## 2.8 Security Considerations

- API key authentication on all endpoints
- TLS termination at NGINX/Traefik layer
- No GPU direct exposure — all traffic through proxy
- Input validation: max image size 10MB, max 20 pages per batch
- Rate limiting: 100 req/min per API key (configurable)
- VRAM watchdog: auto-reject requests if VRAM > 95%

---

# PART 3 — SCALING ROADMAP

| Phase | Configuration | Capacity |
|---|---|---|
| **Phase 1** (Current) | Single A100 80GB, 3 models | ~50 concurrent users |
| **Phase 2** | 2× A100, models split across GPUs | ~200 concurrent users |
| **Phase 3** | K8s cluster, horizontal pod autoscaling | ~1000+ concurrent users |
| **Phase 4** | Multi-region, model replicas, CDN | Global scale |

### Phase 2 Optimization Options
- Move Qwen2.5-7B to FP16 (better quality) on dedicated GPU
- Run ColPali + Qwen3-VL on first GPU with more KV-cache room
- Enable tensor parallelism for Qwen2.5 across both GPUs


## RunPod settings

TomoroAI/tomoro-ai-colqwen3-embed-8b-awq --host 0.0.0.0 --port 8001 --dtype auto --enforce-eager --gpu-memory-utilization 0.95 --max-model-len 32768 --trust-remote-code

---

*Document version: 1.0 | Author: System Architect | Date: Feb 2026*
