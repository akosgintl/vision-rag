"""Tests for MinIO storage service."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from proxy.services.storage import MinioStorage


@pytest.fixture
def storage():
    """MinioStorage with a mocked minio client."""
    with patch("proxy.services.storage.Minio") as MockMinio:
        mock_client = MagicMock()
        MockMinio.return_value = mock_client
        svc = MinioStorage()
        svc.client = mock_client
        return svc


class TestMinioStorage:
    def test_object_path(self, storage):
        path = storage._object_path("my-collection", "doc-123", 5)
        assert path == "my-collection/doc-123/page_5.png"

    def test_ensure_bucket_creates_when_missing(self, storage):
        storage.client.bucket_exists.return_value = False
        storage.ensure_bucket()
        storage.client.make_bucket.assert_called_once_with(storage.bucket)

    def test_ensure_bucket_skips_when_exists(self, storage):
        storage.client.bucket_exists.return_value = True
        storage.ensure_bucket()
        storage.client.make_bucket.assert_not_called()

    async def test_store_page_image(self, storage):
        img_data = b"fake-png-data"
        img_b64 = base64.b64encode(img_data).decode()

        result = await storage.store_page_image(img_b64, "col", "doc-1", 1)
        assert result == "col/doc-1/page_1.png"
        storage.client.put_object.assert_called_once()

    async def test_store_page_images(self, storage):
        images = [base64.b64encode(b"img1").decode(), base64.b64encode(b"img2").decode()]
        count = await storage.store_page_images(images, "col", "doc-1")
        assert count == 2
        assert storage.client.put_object.call_count == 2

    async def test_store_page_images_handles_failures(self, storage):
        from minio.error import S3Error

        images = [base64.b64encode(b"img1").decode(), base64.b64encode(b"img2").decode()]
        storage.client.put_object.side_effect = [
            None,
            S3Error("PutObject", "bucket", "body", "host", "request_id", "resource", "response"),
        ]
        count = await storage.store_page_images(images, "col", "doc-1")
        assert count == 1  # First succeeds, second fails

    async def test_fetch_page_image(self, storage):
        mock_response = MagicMock()
        mock_response.read.return_value = b"fake-png-data"
        storage.client.get_object.return_value = mock_response

        result = await storage.fetch_page_image("col", "doc-1", 1)
        assert result == base64.b64encode(b"fake-png-data").decode()
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()
