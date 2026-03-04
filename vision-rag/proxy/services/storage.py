"""
MinIO/S3 object storage service for page images.

Stores and retrieves rasterized PDF page images at:
    bucket/collection/document_id/page_{N}.png
"""

import asyncio
import base64
import io
from functools import partial

import structlog
from minio import Minio
from minio.error import S3Error

from proxy.config import settings

logger = structlog.get_logger()


class MinioStorage:
    """Wraps the synchronous minio SDK for use in async context."""

    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist. Call once at startup."""
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info("minio_bucket_created", bucket=self.bucket)
        else:
            logger.info("minio_bucket_exists", bucket=self.bucket)

    def _object_path(self, collection: str, document_id: str, page_number: int) -> str:
        return f"{collection}/{document_id}/page_{page_number}.png"

    async def store_page_image(
        self,
        image_b64: str,
        collection: str,
        document_id: str,
        page_number: int,
    ) -> str:
        """
        Store a base64-encoded page image in MinIO.

        Returns the object path.
        """
        img_bytes = base64.b64decode(image_b64)
        path = self._object_path(collection, document_id, page_number)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.client.put_object,
                self.bucket,
                path,
                io.BytesIO(img_bytes),
                len(img_bytes),
                content_type="image/png",
            ),
        )
        return path

    async def store_page_images(
        self,
        page_images: list[str],
        collection: str,
        document_id: str,
    ) -> int:
        """
        Store multiple page images. Returns count of successfully stored images.
        """
        stored = 0
        for page_num, img_b64 in enumerate(page_images, start=1):
            try:
                await self.store_page_image(img_b64, collection, document_id, page_num)
                stored += 1
            except S3Error as e:
                logger.error(
                    "minio_store_failed",
                    document_id=document_id,
                    page=page_num,
                    error=str(e),
                )
        logger.info(
            "images_stored",
            document_id=document_id,
            stored=stored,
            total=len(page_images),
        )
        return stored

    async def fetch_page_image(
        self,
        collection: str,
        document_id: str,
        page_number: int,
    ) -> str:
        """
        Fetch a page image from MinIO.

        Returns base64-encoded image string.
        Raises S3Error if the object doesn't exist.
        """
        path = self._object_path(collection, document_id, page_number)

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            partial(self.client.get_object, self.bucket, path),
        )
        try:
            img_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        return base64.b64encode(img_bytes).decode("utf-8")
