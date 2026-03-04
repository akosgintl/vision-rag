"""Tests for the health check service."""

import httpx
import pytest
import respx

from proxy.services.health import HealthService


@pytest.fixture
def health_service():
    client = httpx.AsyncClient()
    return HealthService(client)


class TestHealthService:
    @respx.mock
    async def test_check_backend_healthy(self, health_service):
        respx.get("http://localhost:8001/health").mock(return_value=httpx.Response(200))
        result = await health_service.check_backend("retrieve")
        assert result.status == "healthy"
        assert result.name == "colpali"
        assert result.latency_ms >= 0

    @respx.mock
    async def test_check_backend_unhealthy(self, health_service):
        respx.get("http://localhost:8001/health").mock(return_value=httpx.Response(503))
        result = await health_service.check_backend("retrieve")
        assert result.status == "unhealthy"

    @respx.mock
    async def test_check_backend_unreachable(self, health_service):
        respx.get("http://localhost:8001/health").mock(side_effect=httpx.ConnectError("refused"))
        result = await health_service.check_backend("retrieve")
        assert result.status == "unreachable"

    @respx.mock
    async def test_check_all_healthy(self, health_service):
        respx.get("http://localhost:8001/health").mock(return_value=httpx.Response(200))
        respx.get("http://localhost:8002/health").mock(return_value=httpx.Response(200))
        respx.get("http://localhost:8003/health").mock(return_value=httpx.Response(200))

        result = await health_service.check_all()
        assert result.status == "ok"
        assert len(result.backends) == 3
        assert all(b.status == "healthy" for b in result.backends)

    @respx.mock
    async def test_check_all_degraded(self, health_service):
        respx.get("http://localhost:8001/health").mock(return_value=httpx.Response(200))
        respx.get("http://localhost:8002/health").mock(return_value=httpx.Response(503))
        respx.get("http://localhost:8003/health").mock(return_value=httpx.Response(200))

        result = await health_service.check_all()
        assert result.status == "degraded"

    @respx.mock
    async def test_check_all_down(self, health_service):
        respx.get("http://localhost:8001/health").mock(side_effect=httpx.ConnectError("refused"))
        respx.get("http://localhost:8002/health").mock(side_effect=httpx.ConnectError("refused"))
        respx.get("http://localhost:8003/health").mock(side_effect=httpx.ConnectError("refused"))

        result = await health_service.check_all()
        assert result.status == "down"

    def test_uptime_seconds(self, health_service):
        assert health_service.uptime_seconds >= 0
