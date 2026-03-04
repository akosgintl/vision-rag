"""
Resilient backend caller — wraps httpx with circuit breaker + retry + timeout.

All model backend calls should go through BackendCaller instead of raw httpx.
"""

import logging

import httpx
import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from proxy.config import settings
from proxy.services.circuit_breaker import CircuitBreakerRegistry

logger = structlog.get_logger()

# Exceptions worth retrying (transient failures)
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.PoolTimeout,
)

# HTTP status codes worth retrying
RETRYABLE_STATUS_CODES = {502, 503, 504, 429}


class BackendError(Exception):
    """Raised when a backend call fails after retries."""

    def __init__(self, backend: str, status_code: int | None, message: str):
        self.backend = backend
        self.status_code = status_code
        self.message = message
        super().__init__(f"{backend}: {message}")


class BackendCaller:
    """
    Resilient HTTP client for model backend calls.

    Combines:
    - Circuit breaker (fast-fail when backend is down)
    - Retry with exponential backoff (recover from transient errors)
    - Per-operation timeout
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        circuit_breakers: CircuitBreakerRegistry,
    ):
        self.client = client
        self.breakers = circuit_breakers

    async def post(
        self,
        backend_name: str,
        url: str,
        json: dict,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> dict:
        """
        POST to a backend with circuit breaker + retry.

        Args:
            backend_name: Logical backend name (e.g. "colpali", "qwen3vl", "qwen25")
            url: Full URL to POST to
            json: JSON body
            timeout: Per-request timeout (defaults to settings.backend_timeout)
            max_retries: Number of retry attempts

        Returns:
            Parsed JSON response dict

        Raises:
            CircuitOpenError: If circuit is open
            BackendError: If all retries are exhausted
        """
        effective_timeout = timeout or settings.backend_timeout
        breaker = self.breakers.get(backend_name)

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=0.5, max=10),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
            reraise=True,
        )
        async def _do_request() -> httpx.Response:
            response = await self.client.post(
                url,
                json=json,
                timeout=effective_timeout,
            )
            # Retry on transient HTTP errors
            if response.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "backend_retryable_status",
                    backend=backend_name,
                    status=response.status_code,
                    url=url,
                )
                raise httpx.ReadTimeout(f"Retryable status {response.status_code}")
            return response

        async with breaker:
            try:
                response = await _do_request()
            except RETRYABLE_EXCEPTIONS as e:
                raise BackendError(backend_name, None, f"Connection failed after {max_retries} retries: {e}") from e

            if response.status_code >= 400:
                detail = response.text[:500]
                raise BackendError(backend_name, response.status_code, f"HTTP {response.status_code}: {detail}")

            return response.json()

    async def stream(
        self,
        backend_name: str,
        url: str,
        json: dict,
        timeout: float | None = None,
    ):
        """
        Streaming POST to a backend with circuit breaker (no retry for streams).

        Yields text chunks. Handles backend disconnects gracefully.
        """
        effective_timeout = timeout or settings.backend_timeout
        breaker = self.breakers.get(backend_name)

        await breaker.check_available()
        try:
            async with self.client.stream(
                "POST",
                url,
                json=json,
                timeout=effective_timeout,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    await breaker.record_failure()
                    raise BackendError(
                        backend_name, response.status_code, f"HTTP {response.status_code}: {body.decode()[:500]}"
                    )
                async for chunk in response.aiter_text():
                    yield chunk
            await breaker.record_success()
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            await breaker.record_failure()
            logger.error("backend_stream_failed", backend=backend_name, error=str(e))
            yield f'data: {{"error": "Backend stream interrupted: {e}"}}\n\n'
        except BackendError:
            raise
        except Exception as e:
            await breaker.record_failure()
            raise BackendError(backend_name, None, f"Stream failed: {e}") from e
