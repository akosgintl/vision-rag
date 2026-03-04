"""Tests for the resilient BackendCaller."""

import httpx
import pytest
import respx

from proxy.services.backend import BackendCaller, BackendError
from proxy.services.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError


@pytest.fixture
def backend():
    client = httpx.AsyncClient()
    breakers = CircuitBreakerRegistry(failure_threshold=3, recovery_timeout=30.0)
    yield BackendCaller(client=client, circuit_breakers=breakers)


class TestBackendPost:
    @respx.mock
    async def test_successful_post(self, backend):
        respx.post("http://model:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": [[0.1]]}]})
        )
        result = await backend.post("colpali", "http://model:8001/v1/embeddings", json={"input": "test"})
        assert result == {"data": [{"embedding": [[0.1]]}]}

    @respx.mock
    async def test_raises_backend_error_on_4xx(self, backend):
        respx.post("http://model:8001/v1/embeddings").mock(return_value=httpx.Response(400, text="Bad request body"))
        with pytest.raises(BackendError) as exc_info:
            await backend.post("colpali", "http://model:8001/v1/embeddings", json={})
        assert exc_info.value.status_code == 400
        assert "Bad request body" in exc_info.value.message

    @respx.mock
    async def test_retries_on_503(self, backend):
        route = respx.post("http://model:8001/test").mock(
            side_effect=[
                httpx.Response(503, text="busy"),
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        result = await backend.post("colpali", "http://model:8001/test", json={}, max_retries=3)
        assert result == {"ok": True}
        assert route.call_count == 3

    @respx.mock
    async def test_exhausts_retries_on_persistent_failure(self, backend):
        respx.post("http://model:8001/test").mock(side_effect=httpx.ConnectTimeout("timeout"))
        with pytest.raises(BackendError) as exc_info:
            await backend.post("colpali", "http://model:8001/test", json={}, max_retries=2)
        assert "Connection failed" in exc_info.value.message

    @respx.mock
    async def test_circuit_opens_after_failures(self, backend):
        respx.post("http://model:8001/test").mock(return_value=httpx.Response(500, text="Internal error"))
        # Exhaust circuit breaker threshold (3 failures)
        for _ in range(3):
            with pytest.raises(BackendError):
                await backend.post("colpali", "http://model:8001/test", json={}, max_retries=1)

        # Next call should fail with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await backend.post("colpali", "http://model:8001/test", json={})


class TestBackendStream:
    @respx.mock
    async def test_stream_yields_chunks(self, backend):
        respx.post("http://model:8003/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="data: chunk1\n\ndata: chunk2\n\n")
        )
        chunks = []
        async for chunk in backend.stream("qwen25", "http://model:8003/v1/chat/completions", json={}):
            chunks.append(chunk)
        assert len(chunks) > 0

    @respx.mock
    async def test_stream_error_on_4xx(self, backend):
        respx.post("http://model:8003/v1/chat/completions").mock(return_value=httpx.Response(400, text="Bad request"))
        with pytest.raises(BackendError) as exc_info:
            async for _ in backend.stream("qwen25", "http://model:8003/v1/chat/completions", json={}):
                pass
        assert exc_info.value.status_code == 400
