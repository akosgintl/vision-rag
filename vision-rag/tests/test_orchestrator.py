"""Tests for the pipeline orchestrator."""

import pytest

from proxy.models.responses import PipelineQueryResponse
from proxy.services.orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator(mock_backend, mock_embedding_index, mock_storage):
    return PipelineOrchestrator(
        backend=mock_backend,
        embedding_index=mock_embedding_index,
        storage=mock_storage,
    )


class TestPipelineOrchestrator:
    async def test_query_no_results(self, orchestrator, mock_embedding_index):
        """When Qdrant returns no results, return a 'no documents found' answer."""
        mock_embedding_index.search.return_value = []

        result = await orchestrator.query(query="something obscure")
        assert isinstance(result, PipelineQueryResponse)
        assert "No relevant documents" in result.answer
        assert result.sources == []

    async def test_query_full_pipeline(self, orchestrator, mock_backend, mock_embedding_index, mock_storage):
        """Test the full retrieve -> extract -> generate flow."""
        # Step 1: ColPali returns query embeddings
        mock_backend.post.side_effect = [
            # encode_query call
            {"data": [{"embedding": [[0.1] * 128]}], "usage": {"total_tokens": 5}},
            # extract_page call
            {"choices": [{"message": {"content": "extracted data"}}], "usage": {"total_tokens": 50}},
            # generate_answer call
            {"choices": [{"message": {"content": "The answer is 42"}}], "usage": {"total_tokens": 30}},
        ]

        # Step 2: Qdrant returns a result
        mock_embedding_index.search.return_value = [
            {"document_id": "doc-1", "page_number": 1, "score": 0.95, "collection": "default"}
        ]

        # Step 3: MinIO returns an image
        mock_storage.fetch_page_image.return_value = "base64img"

        result = await orchestrator.query(query="What is the answer?")
        assert isinstance(result, PipelineQueryResponse)
        assert result.answer == "The answer is 42"
        assert len(result.sources) == 1
        assert result.sources[0].document_id == "doc-1"
        assert result.tokens.retrieval_tokens == 5
        assert result.tokens.extraction_tokens == 50
        assert result.tokens.generation_tokens == 30

    async def test_query_skips_unavailable_images(self, orchestrator, mock_backend, mock_embedding_index, mock_storage):
        """When MinIO fetch fails, extraction is skipped for that page."""
        mock_backend.post.side_effect = [
            {"data": [{"embedding": [[0.1] * 128]}], "usage": {"total_tokens": 5}},
            # generate_answer call (no extraction because image unavailable)
            {"choices": [{"message": {"content": "Partial answer"}}], "usage": {"total_tokens": 20}},
        ]

        mock_embedding_index.search.return_value = [
            {"document_id": "doc-1", "page_number": 1, "score": 0.9, "collection": "default"}
        ]
        mock_storage.fetch_page_image.side_effect = Exception("MinIO down")

        result = await orchestrator.query(query="test", include_extractions=True)
        assert "Partial answer" in result.answer
        assert any("unavailable" in e.lower() for e in result.extractions)

    def test_build_context(self, orchestrator):
        from proxy.models.responses import SourceReference

        sources = [
            SourceReference(document_id="d1", page_number=1, relevance_score=0.9, collection="default"),
            SourceReference(document_id="d2", page_number=3, relevance_score=0.7, collection="default"),
        ]
        context = orchestrator._build_context(["text1", "text2"], sources)
        assert "Page 1" in context
        assert "Page 3" in context
        assert "text1" in context
        assert "---" in context
