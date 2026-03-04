"""Tests for the centralized prompts module."""

import re

from proxy.prompts import (
    PROMPT_VERSION,
    ExtractionPrompts,
    GenerationPrompts,
    SummarizationPrompts,
)


class TestPromptVersion:
    def test_semver_format(self):
        assert re.match(r"^\d+\.\d+\.\d+$", PROMPT_VERSION)


class TestExtractionPrompts:
    def test_default_mentions_json(self):
        assert "JSON" in ExtractionPrompts.DEFAULT

    def test_default_mentions_expected_keys(self):
        for key in ("tables", "key_values", "text_blocks"):
            assert key in ExtractionPrompts.DEFAULT

    def test_query_contextual_has_placeholder(self):
        assert "{query}" in ExtractionPrompts.QUERY_CONTEXTUAL

    def test_format_query_contextual(self):
        result = ExtractionPrompts.format_query_contextual("revenue in Q3")
        assert "revenue in Q3" in result
        assert "{query}" not in result

    def test_format_query_contextual_mentions_json(self):
        result = ExtractionPrompts.format_query_contextual("test")
        assert "JSON" in result


class TestSummarizationPrompts:
    def test_all_styles_present(self):
        assert set(SummarizationPrompts.STYLES.keys()) == {
            "concise",
            "detailed",
            "bullet_points",
        }

    def test_format_system_known_style(self):
        result = SummarizationPrompts.format_system("detailed")
        assert "detailed" in result
        assert "{style_instruction}" not in result

    def test_format_system_unknown_style_falls_back_to_concise(self):
        result = SummarizationPrompts.format_system("nonexistent")
        assert SummarizationPrompts.STYLES["concise"] in result

    def test_format_user(self):
        result = SummarizationPrompts.format_user("Some document text here.")
        assert "Some document text here." in result
        assert "{content}" not in result

    def test_system_template_has_placeholder(self):
        assert "{style_instruction}" in SummarizationPrompts.SYSTEM

    def test_user_template_has_placeholder(self):
        assert "{content}" in SummarizationPrompts.USER


class TestGenerationPrompts:
    def test_system_mentions_document_analysis(self):
        assert "document analysis" in GenerationPrompts.SYSTEM.lower()

    def test_system_mentions_cite_pages(self):
        assert "page numbers" in GenerationPrompts.SYSTEM

    def test_user_template_has_placeholders(self):
        assert "{context}" in GenerationPrompts.USER
        assert "{query}" in GenerationPrompts.USER

    def test_format_user(self):
        result = GenerationPrompts.format_user(
            context="Page 1: Revenue was $10M.",
            query="What was the revenue?",
        )
        assert "Page 1: Revenue was $10M." in result
        assert "What was the revenue?" in result
        assert "{context}" not in result
        assert "{query}" not in result
