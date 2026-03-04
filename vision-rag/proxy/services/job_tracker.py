"""
Redis-backed job status tracker for async ingestion jobs.

Stores job state as a Redis hash with TTL for automatic cleanup.
"""

from datetime import UTC, datetime

import redis.asyncio as aioredis
import structlog

from proxy.config import settings

logger = structlog.get_logger()

JOB_TTL_SECONDS = 86400 * 7  # 7 days


class JobTracker:
    """Tracks async ingestion job status via Redis."""

    def __init__(self):
        self.redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    def _key(self, job_id: str) -> str:
        return f"vrag:job:{job_id}"

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def create_job(
        self,
        job_id: str,
        collection: str,
        document_url: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Create a new pending job."""
        now = datetime.now(UTC).isoformat()
        job = {
            "job_id": job_id,
            "status": "pending",
            "collection": collection,
            "document_url": document_url or "",
            "filename": filename or "",
            "progress": "0",
            "total_pages": "0",
            "processed_pages": "0",
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        key = self._key(job_id)
        await self.redis.hset(key, mapping=job)
        await self.redis.expire(key, JOB_TTL_SECONDS)
        logger.info("job_created", job_id=job_id, collection=collection)
        return job

    async def update_progress(
        self,
        job_id: str,
        processed_pages: int,
        total_pages: int,
    ) -> None:
        """Update job progress during ingestion."""
        now = datetime.now(UTC).isoformat()
        progress = str(round(processed_pages / total_pages, 4)) if total_pages > 0 else "0"
        await self.redis.hset(
            self._key(job_id),
            mapping={
                "status": "processing",
                "processed_pages": str(processed_pages),
                "total_pages": str(total_pages),
                "progress": progress,
                "updated_at": now,
            },
        )

    async def complete_job(
        self,
        job_id: str,
        total_pages: int,
        indexed_pages: int,
    ) -> None:
        """Mark a job as completed."""
        now = datetime.now(UTC).isoformat()
        status = "completed" if indexed_pages == total_pages else "partial"
        await self.redis.hset(
            self._key(job_id),
            mapping={
                "status": status,
                "total_pages": str(total_pages),
                "processed_pages": str(indexed_pages),
                "progress": "1.0",
                "updated_at": now,
            },
        )
        logger.info("job_completed", job_id=job_id, status=status, pages=f"{indexed_pages}/{total_pages}")

    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        now = datetime.now(UTC).isoformat()
        await self.redis.hset(
            self._key(job_id),
            mapping={
                "status": "failed",
                "error": error[:1000],
                "updated_at": now,
            },
        )
        logger.error("job_failed", job_id=job_id, error=error[:200])

    async def get_status(self, job_id: str) -> dict | None:
        """
        Get job status. Returns None if job doesn't exist.

        Numeric fields are converted back from their Redis string representation.
        """
        data = await self.redis.hgetall(self._key(job_id))
        if not data:
            return None

        return {
            "job_id": data.get("job_id", job_id),
            "status": data.get("status", "unknown"),
            "collection": data.get("collection", ""),
            "progress": float(data.get("progress", 0)),
            "total_pages": int(data.get("total_pages", 0)),
            "processed_pages": int(data.get("processed_pages", 0)),
            "error": data.get("error", "") or None,
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
        }

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()
