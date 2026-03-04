"""Health check service for all backends and infrastructure dependencies."""

import asyncio
import time

import httpx
import structlog

from proxy.config import settings
from proxy.models.responses import BackendStatus, HealthResponse

logger = structlog.get_logger()


class HealthService:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.start_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 1)

    async def check_backend(self, name: str) -> BackendStatus:
        """Check health of a single model backend."""
        backend = settings.backends[name]
        start = time.time()
        try:
            resp = await self.client.get(
                f"{backend.url}{backend.health_path}",
                timeout=settings.health_check_timeout,
            )
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return BackendStatus(
                    name=backend.name,
                    status="healthy",
                    latency_ms=round(latency, 2),
                    model_id=backend.model_id,
                )
            else:
                logger.warning("backend_unhealthy", name=name, status=resp.status_code)
                return BackendStatus(
                    name=backend.name,
                    status="unhealthy",
                    latency_ms=round(latency, 2),
                    model_id=backend.model_id,
                )
        except Exception as e:
            logger.error("backend_unreachable", name=name, error=str(e))
            return BackendStatus(
                name=backend.name,
                status="unreachable",
                model_id=backend.model_id,
            )

    async def check_all(self) -> HealthResponse:
        """Check health of all model backends concurrently."""
        tasks = [self.check_backend(name) for name in settings.backends]
        statuses = await asyncio.gather(*tasks)

        healthy_count = sum(1 for s in statuses if s.status == "healthy")
        total = len(statuses)

        if healthy_count == total:
            overall = "ok"
        elif healthy_count > 0:
            overall = "degraded"
        else:
            overall = "down"

        return HealthResponse(
            status=overall,
            backends=list(statuses),
            uptime_seconds=self.uptime_seconds,
            version=settings.app_version,
        )

    async def check_infra(self) -> dict:
        """Check all infrastructure dependencies for readiness."""
        checks = await asyncio.gather(
            self._check_qdrant(),
            self._check_redis(),
            self._check_minio(),
            self._check_postgres(),
            return_exceptions=True,
        )
        names = ["qdrant", "redis", "minio", "postgres"]
        results = {}
        for name, result in zip(names, checks, strict=False):
            if isinstance(result, Exception):
                results[name] = {"status": "unreachable", "error": str(result)}
            else:
                results[name] = result
        return results

    async def readiness(self) -> dict:
        """
        Full readiness check: model backends + infrastructure.

        Returns {ready: bool, backends: {...}, infra: {...}}
        Used by /readyz for Kubernetes readiness probes.
        """
        backend_result = await self.check_all()
        infra_result = await self.check_infra()

        backends_ready = backend_result.status in ("ok", "degraded")
        infra_ready = all(v.get("status") == "healthy" for v in infra_result.values())

        return {
            "ready": backends_ready and infra_ready,
            "status": "ready" if (backends_ready and infra_ready) else "not_ready",
            "backends": {
                "status": backend_result.status,
                "details": [b.model_dump() for b in backend_result.backends],
            },
            "infra": infra_result,
            "uptime_seconds": self.uptime_seconds,
            "version": settings.app_version,
        }

    async def _check_qdrant(self) -> dict:
        """Check Qdrant connectivity."""
        start = time.time()
        try:
            resp = await self.client.get(
                f"{settings.qdrant_url}/healthz",
                timeout=settings.health_check_timeout,
            )
            latency = round((time.time() - start) * 1000, 2)
            return {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "latency_ms": latency,
            }
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def _check_redis(self) -> dict:
        """Check Redis connectivity via the job tracker."""
        import redis.asyncio as aioredis

        start = time.time()
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            pong = await asyncio.wait_for(r.ping(), timeout=settings.health_check_timeout)
            latency = round((time.time() - start) * 1000, 2)
            await r.aclose()
            return {
                "status": "healthy" if pong else "unhealthy",
                "latency_ms": latency,
            }
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def _check_minio(self) -> dict:
        """Check MinIO connectivity."""
        start = time.time()
        try:
            # MinIO health endpoint
            resp = await self.client.get(
                f"http://{settings.minio_endpoint}/minio/health/live",
                timeout=settings.health_check_timeout,
            )
            latency = round((time.time() - start) * 1000, 2)
            return {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "latency_ms": latency,
            }
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def _check_postgres(self) -> dict:
        """Check PostgreSQL connectivity."""

        start = time.time()
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._pg_ping)
            latency = round((time.time() - start) * 1000, 2)
            return {"status": "healthy", "latency_ms": latency}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    @staticmethod
    def _pg_ping():
        """Synchronous PostgreSQL ping."""
        import psycopg2

        conn = psycopg2.connect(dsn=settings.postgres_dsn, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()


async def wait_for_backends(client: httpx.AsyncClient, max_retries: int = 60, interval: float = 5.0):
    """Wait until all backends are healthy. Used during startup."""
    health = HealthService(client)
    for attempt in range(max_retries):
        result = await health.check_all()
        if result.status == "ok":
            logger.info("all_backends_healthy", attempt=attempt + 1)
            return
        logger.info(
            "waiting_for_backends",
            attempt=attempt + 1,
            status=result.status,
            backends={b.name: b.status for b in result.backends},
        )
        await asyncio.sleep(interval)

    raise RuntimeError(f"Backends not ready after {max_retries * interval}s")
