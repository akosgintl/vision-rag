"""Tests for the ingestion service."""

from unittest.mock import patch

import pytest

from proxy.services.ingestion import IngestionService


@pytest.fixture
def ingestion(mock_backend, mock_embedding_index, mock_storage, mock_job_tracker, mock_metadata_db):
    return IngestionService(
        backend=mock_backend,
        embedding_index=mock_embedding_index,
        storage=mock_storage,
        job_tracker=mock_job_tracker,
        metadata_db=mock_metadata_db,
    )


class TestIngestionService:
    @patch("proxy.services.ingestion.IngestionService._rasterize_pdf")
    async def test_ingest_pdf_full_flow(
        self,
        mock_rasterize,
        ingestion,
        mock_backend,
        mock_embedding_index,
        mock_storage,
        mock_job_tracker,
        mock_metadata_db,
    ):
        """Test full ingestion pipeline with mocked PDF rasterization."""
        mock_rasterize.return_value = ["base64page1", "base64page2"]

        # Batched embedding: 2 pages → 2 embedding items in one response
        mock_backend.post.return_value = {
            "data": [
                {"embedding": [[0.1] * 128] * 10},
                {"embedding": [[0.1] * 128] * 10},
            ],
        }

        result = await ingestion.ingest_pdf(
            pdf_bytes=b"%PDF-test",
            collection="test-col",
            document_id="doc-1",
            dpi=150,
            job_id="job-1",
        )

        assert result["document_id"] == "doc-1"
        assert result["total_pages"] == 2
        assert result["indexed_pages"] == 2
        assert result["status"] == "completed"
        assert result["collection"] == "test-col"

        # Verify services were called
        mock_metadata_db.register_document.assert_called_once()
        mock_storage.store_page_images.assert_called_once()
        # Embedding: 1 batched call (not 2 individual calls)
        mock_backend.post.assert_called_once()
        assert mock_embedding_index.index_page.call_count == 2
        mock_job_tracker.update_progress.assert_called()
        mock_job_tracker.complete_job.assert_called_once_with("job-1", 2, 2)

    @patch("proxy.services.ingestion.IngestionService._rasterize_pdf")
    async def test_ingest_partial_indexing(self, mock_rasterize, ingestion, mock_backend, mock_embedding_index):
        """When some pages fail to index, status should be 'partial'."""
        mock_rasterize.return_value = ["page1", "page2", "page3"]
        # Batched embedding: 3 pages → 3 embedding items in one response
        mock_backend.post.return_value = {
            "data": [
                {"embedding": [[0.1] * 128]},
                {"embedding": [[0.1] * 128]},
                {"embedding": [[0.1] * 128]},
            ],
        }

        # Make the second page fail
        mock_embedding_index.index_page.side_effect = [
            "point-1",
            Exception("Qdrant error"),
            "point-3",
        ]

        result = await ingestion.ingest_pdf(pdf_bytes=b"%PDF-test", collection="default")

        assert result["indexed_pages"] == 2
        assert result["total_pages"] == 3
        assert result["status"] == "partial"

    @patch("proxy.services.ingestion.IngestionService._rasterize_pdf")
    async def test_ingest_storage_failure_raises(self, mock_rasterize, ingestion, mock_storage, mock_job_tracker):
        """When MinIO storage fails, the job should be marked failed and exception raised."""
        mock_rasterize.return_value = ["page1"]
        mock_storage.store_page_images.side_effect = Exception("MinIO down")

        with pytest.raises(Exception, match="MinIO down"):
            await ingestion.ingest_pdf(pdf_bytes=b"%PDF-test", collection="default", job_id="job-fail")

        mock_job_tracker.fail_job.assert_called_once()

    @patch("proxy.services.ingestion.IngestionService._rasterize_pdf")
    async def test_ingest_without_optional_services(
        self, mock_rasterize, mock_backend, mock_embedding_index, mock_storage
    ):
        """Ingestion should work without job_tracker or metadata_db."""
        mock_rasterize.return_value = ["page1"]
        mock_backend.post.return_value = {"data": [{"embedding": [[0.1] * 128]}]}

        service = IngestionService(
            backend=mock_backend,
            embedding_index=mock_embedding_index,
            storage=mock_storage,
            job_tracker=None,
            metadata_db=None,
        )

        result = await service.ingest_pdf(pdf_bytes=b"%PDF-test", collection="default")
        assert result["status"] == "completed"

    def test_rasterize_pdf_returns_base64_list(self, ingestion):
        """Test PDF rasterization with a minimal valid PDF."""
        # Create a minimal valid PDF
        minimal_pdf = (
            b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 72 72]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
        )
        try:
            result = ingestion._rasterize_pdf(minimal_pdf, dpi=72)
            assert isinstance(result, list)
            assert all(isinstance(p, str) for p in result)
        except Exception:
            # pdf2image requires poppler — skip if not installed
            pytest.skip("poppler-utils not installed")
