"""Tests for Pydantic request model validation."""

import pytest
from pydantic import ValidationError

from proxy.models.requests import (
    MAX_QUERY_LENGTH,
    ExtractPageRequest,
    GenerateChatRequest,
    GenerateSummarizeRequest,
    PipelineQueryRequest,
    RetrievalIndexRequest,
    RetrievalSearchRequest,
)


class TestRetrievalSearchRequest:
    def test_valid_request(self):
        req = RetrievalSearchRequest(query="find invoices")
        assert req.query == "find invoices"
        assert req.collection == "default"
        assert req.top_k == 5

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalSearchRequest(query="")

    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            RetrievalSearchRequest(query="x" * (MAX_QUERY_LENGTH + 1))

    def test_top_k_bounds(self):
        assert RetrievalSearchRequest(query="test", top_k=1).top_k == 1
        assert RetrievalSearchRequest(query="test", top_k=50).top_k == 50
        with pytest.raises(ValidationError):
            RetrievalSearchRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            RetrievalSearchRequest(query="test", top_k=51)

    def test_collection_constraints(self):
        with pytest.raises(ValidationError):
            RetrievalSearchRequest(query="test", collection="")
        with pytest.raises(ValidationError):
            RetrievalSearchRequest(query="test", collection="x" * 129)


class TestRetrievalIndexRequest:
    def test_valid_request(self):
        req = RetrievalIndexRequest(
            document_id="doc-1",
            page_number=1,
            image_base64="aGVsbG8=",
        )
        assert req.document_id == "doc-1"
        assert req.page_number == 1

    def test_page_number_bounds(self):
        with pytest.raises(ValidationError):
            RetrievalIndexRequest(document_id="d", page_number=-1, image_base64="x")
        with pytest.raises(ValidationError):
            RetrievalIndexRequest(document_id="d", page_number=10001, image_base64="x")


class TestExtractPageRequest:
    def test_defaults(self):
        req = ExtractPageRequest(image_base64="aGVsbG8=")
        assert req.output_format == "json"
        assert req.max_tokens == 2048
        assert req.temperature == 0.1

    def test_valid_output_formats(self):
        for fmt in ("json", "markdown", "text"):
            req = ExtractPageRequest(image_base64="x", output_format=fmt)
            assert req.output_format == fmt

    def test_invalid_output_format(self):
        with pytest.raises(ValidationError):
            ExtractPageRequest(image_base64="x", output_format="xml")

    def test_temperature_bounds(self):
        ExtractPageRequest(image_base64="x", temperature=0.0)
        ExtractPageRequest(image_base64="x", temperature=2.0)
        with pytest.raises(ValidationError):
            ExtractPageRequest(image_base64="x", temperature=-0.1)
        with pytest.raises(ValidationError):
            ExtractPageRequest(image_base64="x", temperature=2.1)


class TestGenerateChatRequest:
    def test_valid_messages(self):
        req = GenerateChatRequest(messages=[{"role": "user", "content": "hello"}])
        assert len(req.messages) == 1

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            GenerateChatRequest(messages=[])

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            GenerateChatRequest(messages=[{"role": "admin", "content": "hi"}])

    def test_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            GenerateChatRequest(messages=[{"role": "user"}])

    def test_missing_role_rejected(self):
        with pytest.raises(ValidationError):
            GenerateChatRequest(messages=[{"content": "hi"}])

    def test_valid_system_user_assistant(self):
        req = GenerateChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        assert len(req.messages) == 3


class TestGenerateSummarizeRequest:
    def test_valid_styles(self):
        for style in ("concise", "detailed", "bullet_points"):
            req = GenerateSummarizeRequest(content="some text", style=style)
            assert req.style == style

    def test_invalid_style(self):
        with pytest.raises(ValidationError):
            GenerateSummarizeRequest(content="text", style="verbose")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            GenerateSummarizeRequest(content="")


class TestPipelineQueryRequest:
    def test_defaults(self):
        req = PipelineQueryRequest(query="what is this about?")
        assert req.collection == "default"
        assert req.top_k == 5
        assert req.include_extractions is True

    def test_top_k_max_20(self):
        PipelineQueryRequest(query="test", top_k=20)
        with pytest.raises(ValidationError):
            PipelineQueryRequest(query="test", top_k=21)
