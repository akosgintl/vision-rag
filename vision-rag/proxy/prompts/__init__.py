"""Centralized LLM prompts for the document understanding pipeline."""

from proxy.prompts.prompts import (
    PROMPT_VERSION,
    ExtractionPrompts,
    GenerationPrompts,
    SummarizationPrompts,
)

__all__ = [
    "PROMPT_VERSION",
    "ExtractionPrompts",
    "GenerationPrompts",
    "SummarizationPrompts",
]
