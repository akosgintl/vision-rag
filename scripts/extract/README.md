# Qwen3-VL Document Extraction Notebook

Structured document extraction using **Qwen3-VL-8B-Instruct** served via a **vLLM HTTP server** (`/v1/chat/completions` API). Extracts text, tables, and figures from images, PDFs, and DOCX files — all traced with **Opik** observability.

Designed to run on [RunPod](https://www.runpod.io/) GPU pods. Mirrors the production architecture in `vision-rag/docker-compose.yml` where Qwen3-VL runs as a standalone container.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Input["Input Documents"]
        IMG["JPG / PNG Image"]
        PDF["PDF Document"]
        DOCX["DOCX Document"]
    end

    subgraph Conversion["Format Conversion"]
        LO["LibreOffice<br/>(headless)"]
        P2I["pdf2image<br/>(Poppler)"]
    end

    subgraph Server["vLLM HTTP Server (:8002)"]
        API["/v1/chat/completions"]
        CB["Continuous Batching"]
        PA["PagedAttention<br/>(KV-cache)"]
    end

    subgraph Extraction["2 Extraction Prompts per Page"]
        T["Content Extraction<br/>→ Markdown + HTML tables"]
        F["Figure Extraction<br/>→ bbox + Description + Crop"]
    end

    subgraph Output["Output"]
        MD["Markdown text"]
        HTML["HTML tables"]
        FIG["Figure descriptions + page images"]
        JSON["JSON export"]
    end

    subgraph Observability["Opik Tracing"]
        TR["Traces & Spans"]
        PM["Prompt Versioning"]
        TK["Token Tracking"]
    end

    IMG --> P2I
    PDF --> P2I
    DOCX --> LO --> P2I
    P2I --> |"PIL Images<br/>(1 per page)"| Server

    API --> CB
    CB --> PA

    Server --> T
    Server --> F

    T --> MD
    T --> HTML
    F --> FIG
    MD & HTML & FIG --> JSON

    Server -.-> TR
    T & F -.-> PM
    TR -.-> TK
```

## Processing Pipeline

```mermaid
flowchart LR
    A["Load Document"] --> B["Convert to<br/>Page Images"]
    B --> C["Encode all images<br/>to base64 (once)"]
    C --> D["Split into batches<br/>(50 pages each)"]
    D --> E["Build 2N conversations<br/>(content + figure)"]
    E --> F["HTTP POST per conversation<br/>to /v1/chat/completions"]
    F --> G["Unpack results<br/>(every 2 = 1 page)"]
    G --> H["Post-process<br/>save page images"]
    H --> I["Export JSON +<br/>save PNGs"]
```

## DOCX Conversion Path

```mermaid
flowchart LR
    DOCX["sample.docx"] --> LO["LibreOffice<br/>--headless<br/>--convert-to pdf"]
    LO --> PDF["sample.pdf<br/>(temp dir)"]
    PDF --> |"optionally saved"| SAVED["output/sample.pdf"]
    PDF --> P2I["pdf2image<br/>(pdftoppm @ 300 DPI)"]
    P2I --> IMGS["PIL Image per page"]
```

## Batched Inference Strategy

Each page generates 2 conversations (content + figure). These are submitted as individual HTTP requests to the vLLM server, which handles concurrency via its internal scheduler.

```mermaid
flowchart TB
    subgraph batch["Batch (50 pages = 100 HTTP requests)"]
        direction LR
        subgraph p1["Page 1"]
            T1["Content prompt"]
            F1["Figure prompt"]
        end
        subgraph p2["Page 2"]
            T2["Content prompt"]
            F2["Figure prompt"]
        end
        subgraph pN["Page N..."]
            TN["Content prompt"]
            FN["Figure prompt"]
        end
    end

    batch --> API["POST /v1/chat/completions<br/>(per conversation)"]

    API --> SCHED["vLLM Scheduler"]

    SCHED --> CB["Continuous Batching<br/>- Dynamically adds/removes sequences<br/>- Overlaps prefill & decode"]
    SCHED --> PA["PagedAttention<br/>- KV-cache in fixed blocks<br/>- Shared across sequences"]

    CB --> OUT["100 results"]
    PA --> OUT

    OUT --> UNPACK["Unpack: every 2 results<br/>= [content, figure]<br/>for one page"]
```

## Per-Page Extraction Detail

```mermaid
flowchart TB
    IMG["Page Image<br/>(base64 PNG)"] --> T["Content Extraction"]
    IMG --> F["Figure Extraction"]

    T --> |"max_tokens=4096<br/>temp=0.1"| TMD["Clean Markdown + HTML Tables<br/>- Headings, lists, bold, italic<br/>- Tables as HTML (thead/tbody/colspan)<br/>- Sentinel: [No content detected]"]

    F --> |"max_tokens=8192<br/>temp=0.1"| FJSON["Grounded Figures<br/>- bbox_2d (0-1000 normalized)<br/>- figure_type, caption, description<br/>- Cropped figure images saved"]
```

## Opik Observability Integration

Since vLLM runs as an HTTP server, auto-instrumentation is unavailable. Manual spans are created for each generation via `opik_client.trace()` / `trace.span()`.

```mermaid
flowchart TB
    GEN["HTTP request to vLLM"] --> TRACE["opik_client.trace()<br/>name: vllm_generation_i<br/>input: prompt (truncated)<br/>metadata: model, batch_index<br/>tags: [vllm, batch]"]

    TRACE --> SPAN["trace.span()<br/>type: llm<br/>model: Qwen3-VL-8B-Instruct<br/>provider: vllm"]

    SPAN --> USAGE["usage:<br/>prompt_tokens<br/>completion_tokens<br/>total_tokens"]

    SPAN --> META["metadata:<br/>temperature<br/>max_tokens<br/>finish_reason"]

    TRACE --> END["trace.end()"]

    subgraph Prompts["Prompt Versioning"]
        PT["content_extraction v4.0"]
        PF["figure_extraction v3.0"]
    end

    Prompts -.-> |"Registered via<br/>opik.Prompt()"| OPIK["Opik Dashboard<br/>http://localhost:5173"]
    END -.-> OPIK
```

## vLLM Server vs Offline Inference

| Aspect | Offline (`vllm.LLM`) | vLLM Server (this notebook) |
|--------|----------------------|-----------------------------|
| Model loading | `vllm.LLM()` in-process | vLLM server handles it |
| API | Direct Python calls | OpenAI-compatible REST API |
| Batch processing | `llm.chat()` with list | vLLM continuous batching + PagedAttention |
| Input format | Python dicts | JSON over HTTP |
| Scaling | Single GPU, single process | Horizontal across GPUs/nodes |
| Use case | Development, experimentation | Production, high throughput |

## Model Selection Guide

```mermaid
quadrantChart
    title Qwen3-VL Model Selection
    x-axis Low VRAM --> High VRAM
    y-axis Lower Quality --> Higher Quality
    quadrant-1 Production `high-end`
    quadrant-2 Best cost-performance
    quadrant-3 Prototyping
    quadrant-4 Overkill for most tasks
    2B-Instruct: [0.15, 0.35]
    4B-Instruct: [0.25, 0.50]
    8B-Instruct: [0.45, 0.80]
    32B-Instruct: [0.80, 0.88]
    235B-MoE: [0.95, 0.95]
```

| Model | VRAM (BF16) | DocVQA | OCRBench | Best For |
|-------|-------------|--------|----------|----------|
| **2B-Instruct** | ~5 GB | ~88% | ~780 | Prototyping, validation |
| **4B-Instruct** | ~10 GB | ~92% | ~840 | Consumer GPUs (RTX 3060) |
| **8B-Instruct** | ~18 GB | ~96% | 896 | **Production sweet spot** |
| **32B-Instruct** | ~64 GB | ~97% | 910 | High quality, needs A100 |
| **235B-A22B (MoE)** | multi-GPU | 97%+ | 920+ | State-of-the-art |

## Output Folder Structure

```
extraction_output/
    <image_stem>/
        page_001.png                  # copy of original image
        <image_stem>_extraction.json  # JSON export

    <pdf_stem>/
        page_001.png                  # rasterized page images
        page_002.png
        <pdf_stem>_extraction.json

    <docx_stem>/
        <docx_stem>.pdf               # intermediate PDF (preserved)
        page_001.png
        page_002.png
        page18_fig1_crop.png          # cropped figure images (per detected figure)
        page19_fig1_crop.png
        <docx_stem>_extraction.json
```

> **Figure crops**: When the figure extraction prompt detects figures with `bbox_2d` coordinates, the notebook crops each figure from the source page image and saves it as `page{N}_fig{M}_crop.png`. These are referenced in the `figures` array of the corresponding page entry in the extraction JSON.

## Quick Start

1. **Create a RunPod pod** with a PyTorch 2.8.0 template and an appropriate GPU (see model table above).

2. **Install system dependencies** (run once):
   ```bash
   apt-get update -qq && apt-get install -y -qq poppler-utils libreoffice libgl1 libglib2.0-0
   ```

3. **Install Python packages**:
   ```bash
   pip install "vllm>=0.16.0" "httpx>=0.27.0" "Pillow>=10.4.0,<13.0" \
       "pdf2image==1.17.0" "tqdm>=4.66.0" "ipywidgets>=8.1.0" "hf_transfer" "opik"
   ```

4. **Start the vLLM server**:
   ```bash
   vllm serve Qwen/Qwen3-VL-8B-Instruct \
       --port 8002 \
       --task generate \
       --trust-remote-code \
       --dtype bfloat16 \
       --max-model-len 16384 \
       --gpu-memory-utilization 0.95 \
       --limit-mm-per-prompt image=10
   ```

5. **Configure Opik** (optional, for tracing):
   ```bash
   git clone https://github.com/comet-ml/opik.git && cd opik
   docker compose --profile opik up -d
   # UI at http://localhost:5173
   ```

6. **Open the notebook**, point `IMAGE_PATH` / `PDF_PATH` / `DOCX_PATH` to your files, and run all cells.

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8002` | vLLM server address |
| `MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `REQUEST_TIMEOUT` | `120.0` | Seconds per HTTP request |
| `DTYPE` | `bfloat16` | Model precision |
| `MAX_MODEL_LEN` | `16384` | Max context length (tokens) |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of VRAM to use |
| `MAX_NEW_TOKENS` | `4096` | Max output tokens (text/table) |
| `FIGURE_MAX_TOKENS` | `8192` | Max output tokens (figures) |
| `TEMPERATURE` | `0.1` | Low = deterministic extraction |
| `PDF_DPI` | `300` | PDF rasterization resolution |
| `BATCH_SIZE` | `50` | Pages per batch (limits KV-cache) |
