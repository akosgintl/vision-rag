"""Centralized LLM prompt definitions for all pipeline stages."""

PROMPT_VERSION = "1.0.0"


class ExtractionPrompts:
    """Prompts for the Qwen3-VL extraction stage."""

    DEFAULT = (
        "Extract all structured data from this document page. "
        "Return as JSON with keys: tables, key_values, text_blocks."
    )

    QUERY_CONTEXTUAL = (
        "Extract all relevant information from this document page "
        "related to: {query}\n\n"
        "Return as structured JSON with keys:\n"
        "- tables: any tables found\n"
        "- key_values: key-value pairs\n"
        "- text_blocks: relevant text passages"
    )

    @staticmethod
    def format_query_contextual(query: str) -> str:
        return ExtractionPrompts.QUERY_CONTEXTUAL.format(query=query)


class SummarizationPrompts:
    """Prompts for the Qwen2.5-7B summarization endpoint."""

    STYLES: dict[str, str] = {
        "concise": "Provide a concise summary in 2-3 sentences.",
        "detailed": "Provide a detailed summary covering all key points.",
        "bullet_points": "Summarize as bullet points, one per key finding.",
    }

    SYSTEM = "You are a document summarization assistant. {style_instruction}"

    USER = "Summarize the following document content:\n\n{content}"

    @staticmethod
    def format_system(style: str) -> str:
        style_instruction = SummarizationPrompts.STYLES.get(style, SummarizationPrompts.STYLES["concise"])
        return SummarizationPrompts.SYSTEM.format(style_instruction=style_instruction)

    @staticmethod
    def format_user(content: str) -> str:
        return SummarizationPrompts.USER.format(content=content)


class GenerationPrompts:
    """Prompts for the Qwen2.5-7B generation stage (pipeline)."""

    SYSTEM = (
        "You are a document analysis assistant. Answer questions "
        "based solely on the provided document extractions. "
        "Cite page numbers when referencing specific information. "
        "If the context doesn't contain enough information, say so."
    )

    USER = "Context:\n{context}\n\nQuestion: {query}"

    @staticmethod
    def format_user(context: str, query: str) -> str:
        return GenerationPrompts.USER.format(context=context, query=query)
