"""Tests for API routers using FastAPI TestClient with mocked backends."""

import base64

from proxy.models.responses import PipelineQueryResponse, SourceReference, TokenUsage


class TestRetrievalRouter:
    def test_search_success(self, client, mock_backend, mock_embedding_index):
        mock_backend.post.return_value = {
            "data": [{"embedding": [[0.1] * 128]}],
            "usage": {"total_tokens": 5},
        }
        mock_embedding_index.search.return_value = [
            {
                "point_id": "p1",
                "document_id": "doc-1",
                "page_number": 1,
                "score": 0.95,
                "collection": "default",
            }
        ]

        resp = client.post("/v1/retrieve/search", json={"query": "find invoices"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["document_id"] == "doc-1"
        assert data["query"] == "find invoices"
        assert data["latency_ms"] >= 0

    def test_search_empty_query_rejected(self, client):
        resp = client.post("/v1/retrieve/search", json={"query": ""})
        assert resp.status_code == 422

    def test_index_page(self, client, mock_backend, mock_embedding_index):
        mock_backend.post.return_value = {
            "data": [{"embedding": [[0.1] * 128] * 10}],
        }

        resp = client.post(
            "/v1/retrieve/index",
            json={
                "document_id": "doc-1",
                "page_number": 1,
                "image_base64": base64.b64encode(b"fake-image").decode(),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "doc-1"
        assert data["status"] == "indexed"


class TestExtractionRouter:
    def test_extract_page(self, client, mock_backend):
        mock_backend.post.return_value = {
            "choices": [{"message": {"content": '{"tables": []}'}}],
            "usage": {"total_tokens": 50},
        }

        resp = client.post(
            "/v1/extract/page",
            json={
                "image_base64": base64.b64encode(b"fake").decode(),
                "prompt": "Extract tables",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == '{"tables": []}'
        assert data["tokens_used"] == 50

    def test_extract_invalid_format(self, client):
        resp = client.post(
            "/v1/extract/page",
            json={
                "image_base64": "x",
                "output_format": "xml",
            },
        )
        assert resp.status_code == 422


class TestGenerationRouter:
    def test_chat(self, client, mock_backend):
        mock_backend.post.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 20},
        }

        resp = client.post(
            "/v1/generate/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Hello!"
        assert data["finish_reason"] == "stop"

    def test_summarize(self, client, mock_backend):
        mock_backend.post.return_value = {
            "choices": [{"message": {"content": "Summary here"}}],
            "usage": {"total_tokens": 30},
        }

        resp = client.post(
            "/v1/generate/summarize",
            json={
                "content": "Long document text here...",
                "style": "concise",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == "Summary here"
        assert data["style"] == "concise"

    def test_invalid_style_rejected(self, client):
        resp = client.post(
            "/v1/generate/summarize",
            json={
                "content": "text",
                "style": "verbose",
            },
        )
        assert resp.status_code == 422


class TestPipelineRouter:
    def test_query_success(self, client, app_with_mocks):
        app_with_mocks.state.orchestrator.query.return_value = PipelineQueryResponse(
            answer="The document discusses...",
            sources=[
                SourceReference(
                    document_id="doc-1",
                    page_number=1,
                    relevance_score=0.9,
                    collection="default",
                )
            ],
            extractions=["extracted text"],
            latency_ms=150.0,
            tokens=TokenUsage(retrieval_tokens=10, extraction_tokens=50, generation_tokens=30),
        )

        resp = client.post("/v1/pipeline/query", json={"query": "What is this about?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The document discusses..."
        assert len(data["sources"]) == 1

    def test_job_status(self, client, mock_job_tracker):
        resp = client.get("/v1/pipeline/job/test-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_job_status_not_found(self, client, mock_job_tracker):
        mock_job_tracker.get_status.return_value = None
        resp = client.get("/v1/pipeline/job/nonexistent")
        assert resp.status_code == 404

    def test_collections_count(self, client, mock_embedding_index):
        mock_embedding_index.count.return_value = 42
        resp = client.get("/v1/pipeline/collections")
        assert resp.status_code == 200
        assert resp.json()["total_pages"] == 42

    def test_documents_list(self, client, mock_metadata_db):
        resp = client.get("/v1/pipeline/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["documents"]) == 1

    def test_document_get(self, client, mock_metadata_db):
        resp = client.get("/v1/pipeline/documents/test-doc")
        assert resp.status_code == 200
        assert resp.json()["document_id"] == "test-doc"

    def test_document_not_found(self, client, mock_metadata_db):
        mock_metadata_db.get_document.return_value = None
        resp = client.get("/v1/pipeline/documents/unknown")
        assert resp.status_code == 404


class TestLivenessEndpoint:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"
