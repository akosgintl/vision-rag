# How Opik Tracing Works in This Code

## What Is Opik?

Opik is an **observability / logging platform for LLM applications**. Think of it as "application performance monitoring" (like Datadog or New Relic) but purpose-built for large-language-model calls. It lets you record every prompt, every response, token counts, latencies, and metadata so you can later search, visualize, and debug your LLM pipeline.

---

## The Two Core Concepts

### 1. Trace

A **trace** represents one end-to-end unit of work — in this case, a single generation request inside a batch.

```python
trace = opik_client.trace(
    name=f"vllm_generation_{i}",
    input={"prompt": prompt_text[:500]},
    metadata={"model": MODEL_ID, "batch_index": i},
    tags=["vllm", "batch"],
)
```

| Parameter   | What it does |
|-------------|-------------|
| `name`      | A human-readable label (here it includes the batch index so you can find it later). |
| `input`     | The data that *entered* this unit of work — the first 500 chars of the prompt. |
| `metadata`  | Arbitrary key-value pairs for filtering/searching (model name, position in the batch). |
| `tags`      | Short labels you can use to filter traces in the Opik dashboard (e.g. show me all "vllm" traces). |

Think of a trace as **opening a folder** — everything that happens inside it will be grouped together.

---

### 2. Span

A **span** is a single operation *inside* a trace. One trace can contain many spans (e.g. retrieval → reranking → LLM call → post-processing). Here there is exactly one span — the LLM call itself.

```python
trace.span(
    name="llm_call",
    type="llm",
    input={"prompt": prompt_text[:500]},
    output={"text": result_text[:1000]},
    model=MODEL_ID,
    provider="vllm",
    usage={
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    },
    metadata={
        "temperature": params_list[i].temperature,
        "max_tokens": params_list[i].max_tokens,
        "finish_reason": finish_reason,
    },
)
```

| Parameter | What it does |
|-----------|-------------|
| `name`    | Label for this step (`"llm_call"`). |
| `type`    | Tells Opik this is an **LLM-type** span, which unlocks special dashboards (token charts, cost tracking, etc.). |
| `input`   | What went *into* the LLM — the prompt (truncated to 500 chars to keep logs manageable). |
| `output`  | What came *out* — the generated text (truncated to 1 000 chars). |
| `model`   | The model identifier so you can compare performance across models. |
| `provider` | The inference backend (`"vllm"` here, could be `"openai"`, `"anthropic"`, etc.). |
| `usage`   | Token counts. Opik uses these to compute cost estimates and spot runaway prompts. |
| `metadata` | Generation parameters and the finish reason (`"stop"`, `"length"`, etc.) for debugging. |

---

## The Lifecycle — Step by Step

```
 Loop iteration i
 ┌──────────────────────────────────────────────┐
 │                                              │
 │  1. Run the vLLM generation (not shown)      │
 │           ↓                                  │
 │  2. trace = opik_client.trace(...)           │  ← open a trace
 │           ↓                                  │
 │  3. trace.span(...)                          │  ← log the LLM call
 │           ↓                                  │
 │  4. trace.end()                              │  ← close & flush
 │                                              │
 └──────────────────────────────────────────────┘
```

1. **Generation happens first** — the model has already produced `result_text`, `prompt_tokens`, `completion_tokens`, and `finish_reason`.
2. **`opik_client.trace(...)`** opens a new trace and records the high-level input/metadata.
3. **`trace.span(...)`** adds a child span with all the details of the LLM call (prompt, output, tokens, params).
4. **`trace.end()`** finalizes the trace and sends everything to the Opik backend (local or cloud).

Because this sits inside a loop over batch index `i`, **one trace + one span is created per generation in the batch**.

---

## What You See in the Opik Dashboard

After the code runs, Opik's UI will show something like:

```
Traces
├─ vllm_generation_0   [vllm] [batch]   prompt_tokens: 128   completion_tokens: 256
│  └─ llm_call (LLM)   model: meta-llama/...   finish_reason: stop
├─ vllm_generation_1   [vllm] [batch]   prompt_tokens: 95    completion_tokens: 312
│  └─ llm_call (LLM)   model: meta-llama/...   finish_reason: stop
├─ vllm_generation_2   ...
│  └─ ...
```

From here you can:

- **Search & filter** by tags, model, finish reason, or any metadata key.
- **Inspect** the full prompt and response for any single call.
- **Track costs** via the token usage numbers.
- **Spot anomalies** — e.g. a generation that consumed far more tokens than expected, or one that ended with `finish_reason: "length"` (meaning it was cut off).

---

## Key Takeaways

| Concept | Analogy |
|---------|---------|
| **Opik Client** | The logger / SDK that talks to the Opik backend. |
| **Trace** | A "folder" grouping all steps of one logical request. |
| **Span** | A single step inside that folder (here, the LLM call). |
| **`type="llm"`** | Tells Opik to treat this span specially — enabling token/cost dashboards. |
| **`trace.end()`** | Closes the folder and flushes the data to the server. |

The overall pattern is: **create a trace → add one or more spans → end the trace**. This gives you full observability into every LLM call your application makes.
