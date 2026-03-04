# Prompt Changelog

## 1.0.0 — 2026-02-19

Initial extraction of all LLM prompts from inline definitions.

- **ExtractionPrompts**: `DEFAULT`, `QUERY_CONTEXTUAL` (from `requests.py`, `orchestrator.py`)
- **SummarizationPrompts**: `STYLES`, `SYSTEM`, `USER` (from `generate.py`)
- **GenerationPrompts**: `SYSTEM`, `USER` (from `orchestrator.py`)
