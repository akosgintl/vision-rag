"""Circuit breaker pattern for backend resilience."""

import asyncio
import time
from enum import Enum

import structlog

logger = structlog.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the circuit is open."""

    def __init__(self, backend: str, retry_after: float):
        self.backend = backend
        self.retry_after = retry_after
        super().__init__(f"Circuit open for {backend}, retry after {retry_after:.0f}s")


class CircuitBreaker:
    """
    Async-safe circuit breaker for a single backend.

    States:
        CLOSED   -> normal, requests pass through
        OPEN     -> backend is down, fast-fail all requests
        HALF_OPEN -> testing recovery with limited requests

    Usage as async context manager:

        async with circuit_breaker:
            result = await backend_call()
        # success/failure recorded automatically
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_attempts = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def is_available(self) -> bool:
        """Whether requests should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_attempts < self.half_open_max
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until the circuit might transition to half-open."""
        if self._state != CircuitState.OPEN:
            return 0
        elapsed = time.time() - self._last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    async def check_available(self) -> None:
        """Raise CircuitOpenError if the circuit is not accepting requests."""
        async with self._lock:
            if self.state == CircuitState.OPEN and not self.is_available:
                raise CircuitOpenError(self.name, self.retry_after)
            # Transition from OPEN -> HALF_OPEN if timeout elapsed
            if self._state == CircuitState.OPEN and self.state == CircuitState.HALF_OPEN:
                logger.info("circuit_half_open", backend=self.name)
                self._state = CircuitState.HALF_OPEN
                self._half_open_attempts = 0

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_attempts += 1
                self._success_count += 1
                if self._half_open_attempts >= self.half_open_max:
                    logger.info("circuit_closed", backend=self.name, after_successes=self._success_count)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)
                self._success_count += 1

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("circuit_reopened", backend=self.name)
                self._state = CircuitState.OPEN
                self._half_open_attempts = 0
            elif self._failure_count >= self.failure_threshold:
                logger.warning(
                    "circuit_opened",
                    backend=self.name,
                    failures=self._failure_count,
                )
                self._state = CircuitState.OPEN

    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_attempts = 0
            logger.info("circuit_reset", backend=self.name)

    async def __aenter__(self):
        await self.check_available()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure()
        # Don't suppress the exception
        return False

    def status_dict(self) -> dict:
        """Return status for health/metrics endpoints."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "retry_after": round(self.retry_after, 1) if self._state == CircuitState.OPEN else None,
        }


class CircuitBreakerRegistry:
    """Manages circuit breakers for all backends."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 2,
    ):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_requests = half_open_requests

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
                half_open_requests=self._half_open_requests,
            )
        return self._breakers[name]

    def status(self) -> dict[str, dict]:
        return {name: cb.status_dict() for name, cb in self._breakers.items()}
