# How Opik Prompt Versioning Works in This Code

## The Problem This Solves

Imagine you have an LLM-powered document extraction pipeline. Over time you **tweak your prompts** — maybe you add a new instruction, change the output format, or fix an edge case. A month later something breaks and you ask yourself:

> "Which version of the prompt was running when this worked correctly?"

Without prompt versioning you're digging through Git history or guessing. **Opik's Prompt registry** solves this by treating prompts like versioned artifacts — similar to how Docker tags container images or how MLflow versions models.

---

## The Core Concept: `opik.Prompt`

```python
text_extraction_prompt = opik.Prompt(
    name="text_extraction",
    prompt=TEXT_EXTRACTION_PROMPT,
    metadata={"task": "text", "version": "1.0"},
)
```

When this line executes, Opik does the following behind the scenes:

```
┌─────────────────────────────────────────────────────┐
│  Does a prompt named "text_extraction" already       │
│  exist in the Opik registry?                         │
│                                                      │
│   NO  ──→  Create it as version 1                    │
│                                                      │
│   YES ──→  Has the prompt text changed?              │
│              │                                       │
│              ├─ NO  ──→  Return the existing version │
│              │           (no duplicate created)       │
│              │                                       │
│              └─ YES ──→  Create a NEW version         │
│                          (auto-incremented)           │
└─────────────────────────────────────────────────────┘
```

This means you can **re-run your script 100 times** with the same prompt text and Opik will not create 100 copies — it recognizes the content is identical and returns the existing version. But the moment you change even one character in `TEXT_EXTRACTION_PROMPT`, Opik automatically creates a new version.

---

## Walking Through the Code

The pipeline registers **three** prompts — one per extraction task:

```python
# 1️⃣  For extracting plain text from documents
text_extraction_prompt = opik.Prompt(
    name="text_extraction",
    prompt=TEXT_EXTRACTION_PROMPT,
    metadata={"task": "text", "version": "1.0"},
)

# 2️⃣  For extracting tables from documents
table_extraction_prompt = opik.Prompt(
    name="table_extraction",
    prompt=TABLE_EXTRACTION_PROMPT,
    metadata={"task": "table", "version": "1.0"},
)

# 3️⃣  For extracting figures/images from documents
figure_extraction_prompt = opik.Prompt(
    name="figure_extraction",
    prompt=FIGURE_EXTRACTION_PROMPT,
    metadata={"task": "figure", "version": "1.0"},
)
```

| Parameter  | What It Does |
|-----------|-------------|
| `name`     | A **unique identifier** in the registry. This is how Opik looks up whether the prompt already exists. Think of it as a dictionary key. |
| `prompt`   | The **actual prompt text** (the template string you send to the LLM). This is what Opik hashes to detect changes between versions. |
| `metadata` | Arbitrary key-value pairs for your own bookkeeping. Opik stores them alongside the version but does **not** use them for change detection. |

---

## What the Opik Registry Looks Like After Running This

```
Opik Prompt Registry
│
├─ text_extraction
│   ├─ Version 1  ──  prompt: "Extract all text from..."      metadata: {task: text, version: 1.0}
│   ├─ Version 2  ──  prompt: "Extract all text from... v2"   metadata: {task: text, version: 2.0}  ← after a future edit
│   └─ ...
│
├─ table_extraction
│   └─ Version 1  ──  prompt: "Identify and extract all..."   metadata: {task: table, version: 1.0}
│
└─ figure_extraction
    └─ Version 1  ──  prompt: "Locate all figures and..."     metadata: {task: figure, version: 1.0}
```

Each prompt name is a **row**. Each content change creates a new **version column**. Old versions are never deleted — you can always go back and inspect them.

---

## How Versioned Prompts Connect to Traces

Once registered, you can reference these prompt objects when logging traces and spans (as shown in the previous tracing guide). The connection looks like this:

```
Trace: process_document_42
├─ Span: text_extraction  ──→  uses prompt "text_extraction" v2
├─ Span: table_extraction ──→  uses prompt "table_extraction" v1
└─ Span: figure_extraction ──→  uses prompt "figure_extraction" v1
```

This link is powerful because you can now **filter the Opik dashboard** to answer questions like:

- "Show me all traces that used `text_extraction` version 1" — useful for comparing quality before and after a prompt change.
- "Which prompt version was active when this bad output was generated?" — useful for root-cause analysis.

---

## A Real-World Scenario

Here's a timeline showing why this matters:

| Week | What Happened | Opik State |
|------|--------------|------------|
| 1 | You deploy the pipeline with the initial prompts. | `text_extraction` v1, `table_extraction` v1, `figure_extraction` v1 |
| 3 | Tables are being extracted poorly, so you rewrite `TABLE_EXTRACTION_PROMPT`. | `table_extraction` auto-increments to **v2**. |
| 4 | Quality improves! But you want proof. | Filter Opik dashboard: compare outputs of `table_extraction` v1 vs v2. |
| 6 | A colleague edits `TEXT_EXTRACTION_PROMPT` without telling you. Something breaks. | `text_extraction` auto-increments to **v2**. You spot the new version in the registry and diff it against v1 to find the problem. |

Without Opik's prompt registry, week 6 would have been a painful debugging session. With it, you open the dashboard, see a new version appeared, diff it, and find the issue in minutes.

---

## Key Takeaways

| Concept | What It Means |
|---------|--------------|
| **`opik.Prompt`** | Registers a prompt in the Opik registry as a versioned artifact. |
| **`name`** | The unique key used to look up the prompt. |
| **`prompt`** | The actual template text. Changes to this trigger a new version. |
| **`metadata`** | Your own labels/tags — stored but not used for versioning logic. |
| **Auto-versioning** | Same text = same version (idempotent). Changed text = new version (automatic). |
| **Immutable history** | Old versions are never overwritten — you always have a full audit trail. |

The overall pattern is: **register your prompts at startup → Opik tracks every change automatically → link prompts to traces for full observability**.
