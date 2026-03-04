"""Tests for the circuit breaker pattern implementation."""

import asyncio

import pytest

from proxy.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    """Unit tests for CircuitBreaker state machine."""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            name="test",
            failure_threshold=3,
            recovery_timeout=0.2,
            half_open_requests=2,
        )

    async def test_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available

    async def test_stays_closed_under_threshold(self, breaker):
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available

    async def test_opens_after_threshold(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert not breaker.is_available

    async def test_open_circuit_raises_error(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.check_available()
        assert exc_info.value.backend == "test"
        assert exc_info.value.retry_after > 0

    async def test_transitions_to_half_open(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.25)
        assert breaker.state == CircuitState.HALF_OPEN

    async def test_half_open_closes_on_success(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        await asyncio.sleep(0.25)

        # Trigger the transition via check_available
        await breaker.check_available()
        assert breaker._state == CircuitState.HALF_OPEN

        # Succeed enough times to close
        await breaker.record_success()
        await breaker.record_success()
        assert breaker._state == CircuitState.CLOSED

    async def test_half_open_reopens_on_failure(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        await asyncio.sleep(0.25)

        await breaker.check_available()
        assert breaker._state == CircuitState.HALF_OPEN

        await breaker.record_failure()
        assert breaker._state == CircuitState.OPEN

    async def test_success_decrements_failure_count(self, breaker):
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker._failure_count == 2

        await breaker.record_success()
        assert breaker._failure_count == 1

    async def test_reset(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        await breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    async def test_context_manager_records_success(self, breaker):
        async with breaker:
            pass
        assert breaker._success_count == 1

    async def test_context_manager_records_failure(self, breaker):
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("boom")
        assert breaker._failure_count == 1

    async def test_context_manager_rejects_when_open(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        with pytest.raises(CircuitOpenError):
            async with breaker:
                pass

    def test_status_dict(self, breaker):
        status = breaker.status_dict()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["retry_after"] is None

    async def test_retry_after_decreases(self, breaker):
        for _ in range(3):
            await breaker.record_failure()
        initial = breaker.retry_after
        await asyncio.sleep(0.05)
        assert breaker.retry_after < initial


class TestCircuitBreakerRegistry:
    def test_creates_breakers_on_demand(self):
        registry = CircuitBreakerRegistry(failure_threshold=3)
        b1 = registry.get("backend-a")
        b2 = registry.get("backend-a")
        assert b1 is b2

    def test_different_names_get_different_breakers(self):
        registry = CircuitBreakerRegistry()
        b1 = registry.get("a")
        b2 = registry.get("b")
        assert b1 is not b2

    def test_status_returns_all(self):
        registry = CircuitBreakerRegistry()
        registry.get("a")
        registry.get("b")
        status = registry.status()
        assert "a" in status
        assert "b" in status
        assert status["a"]["state"] == "closed"
