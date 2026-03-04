# 🧠 Visual-RAG AI: Multi-Model Document Understanding System

A production-ready pipeline for document understanding at scale, running three specialized models on a **single NVIDIA A100 80GB GPU** via vLLM.

## Architecture

```
Client → FastAPI Proxy (:8000)
              ├── /v1/retrieve/*  → ColPali (TomoroAI/tomoro-ai-colqwen3-embed-8b-awq)  :8001  [~8 GB]
              ├── /v1/extract/*   → Qwen3-VL-2B-Instruct                                :8002  [~6 GB]
              ├── /v1/generate/*  → Qwen2.5-7B-Instruct-AWQ                             :8003  [~14 GB]
              └── /v1/pipeline/*  → Orchestrator (all 3 models)
                                                         Total: ~28 GB / 80 GB
```

## Models

| Model | Role | VRAM | Quantization |
|---|---|---|---|
| ColPali (TomoroAI/tomoro-ai-colqwen3-embed-8b-awq) | Visual page retrieval | ~8 GB | AWQ W4A16 |
| Qwen3-VL-2B-Instruct | Structured data extraction | ~6 GB | FP16 |
| Qwen2.5-7B-Instruct-AWQ | RAG answer generation | ~14 GB | AWQ 4-bit |

## Quick Start

### 1. Prerequisites

- NVIDIA A100 80GB GPU (or equivalent)
- Docker & Docker Compose with NVIDIA Container Toolkit
- HuggingFace token (for model downloads)

### 2. Setup

```bash
# Clone and configure
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Start everything
docker compose up -d

# Watch logs
docker compose logs -f proxy
```

### 3. Test

```bash
# Health check
curl http://localhost:8000/health

# Search documents
curl -X POST http://localhost:8000/v1/retrieve/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quarterly revenue", "collection": "default", "top_k": 5}'

# Extract from a page image
curl -X POST http://localhost:8000/v1/extract/page \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64_image>", "prompt": "Extract all tables"}'

# Chat with context
curl -X POST http://localhost:8000/v1/generate/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a document analyst."},
      {"role": "user", "content": "Summarize the key findings."}
    ]
  }'

# Full pipeline query
curl -X POST http://localhost:8000/v1/pipeline/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was our revenue in Q3?", "collection": "default"}'

# Upload and ingest a PDF
curl -X POST http://localhost:8000/v1/pipeline/ingest/upload \
  -F "file=@report.pdf" \
  -F "collection=financial"
```

### 4. Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM model servers
chmod +x scripts/start_models.sh
./scripts/start_models.sh

# In another terminal, start the proxy
uvicorn proxy.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check all backends |
| `/v1/retrieve/search` | POST | Search pages via ColPali |
| `/v1/retrieve/index` | POST | Index a page embedding |
| `/v1/extract/page` | POST | Extract from single page |
| `/v1/extract/batch` | POST | Extract from multiple pages |
| `/v1/generate/chat` | POST | Chat completion |
| `/v1/generate/summarize` | POST | Summarize content |
| `/v1/pipeline/query` | POST | Full pipeline (retrieve→extract→generate) |
| `/v1/pipeline/ingest/upload` | POST | Upload and ingest PDF |
| `/v1/proxy/{service}/{path}` | ANY | Raw passthrough to vLLM |
| `/docs` | GET | OpenAPI documentation |

## Infrastructure

| Service | Port | Purpose |
|---|---|---|
| FastAPI Proxy | 8000 | API gateway |
| ColPali vLLM | 8001 | Retrieval model |
| Qwen3-VL vLLM | 8002 | Extraction model |
| Qwen2.5 vLLM | 8003 | Generation model |
| Qdrant | 6333 | Vector store |
| PostgreSQL | 5432 | Metadata store |
| MinIO | 9000/9001 | Object storage |
| Redis | 6379 | Task queue |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Dashboards |

## Project Structure

```
vision-rag/
├── proxy/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Settings (env vars)
│   ├── routers/
│   │   ├── retrieve.py         # ColPali endpoints
│   │   ├── extract.py          # Qwen3-VL endpoints
│   │   ├── generate.py         # Qwen2.5 endpoints
│   │   └── pipeline.py         # Orchestrated pipeline
│   ├── services/
│   │   ├── health.py           # Health checks
│   │   ├── circuit_breaker.py  # Resilience
│   │   ├── embedding_index.py  # Qdrant integration
│   │   ├── orchestrator.py     # Pipeline logic
│   │   └── ingestion.py        # PDF ingestion
│   ├── middleware/
│   │   ├── rate_limiter.py     # Rate limiting
│   │   └── auth.py             # API key auth
│   └── models/
│       ├── requests.py         # Request schemas
│       └── responses.py        # Response schemas
├── scripts/
│   └── start_models.sh         # Local vLLM launcher
├── monitoring/
│   └── prometheus.yml          # Prometheus config
├── docker-compose.yml          # Full stack deployment
├── Dockerfile                  # Proxy container
├── lmcache_config.yaml         # KV-cache config
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

## License

MIT
