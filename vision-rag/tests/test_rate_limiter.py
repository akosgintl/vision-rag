"""Tests for the Redis-backed rate limiter middleware."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from proxy.middleware.rate_limiter import SLIDING_WINDOW_SCRIPT, RateLimiterMiddleware


class TestRateLimiterMiddleware:
    @pytest.fixture
    def app_with_limiter(self, fake_redis):
        """Create an app with rate limiter backed by fakeredis."""
        app = FastAPI()

        with (
            patch("proxy.middleware.rate_limiter.settings") as mock_settings,
        ):
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests = 3
            mock_settings.rate_limit_window_seconds = 60
            mock_settings.redis_url = "redis://fake:6379/0"

            app.add_middleware(RateLimiterMiddleware, rate=3, window=60)

            @app.get("/test")
            async def test_endpoint():
                return {"ok": True}

            @app.get("/health")
            async def health():
                return {"status": "ok"}

        return app, mock_settings

    def test_skips_health_endpoint(self, app_with_limiter):
        app, mock_settings = app_with_limiter
        client = TestClient(app)
        with patch("proxy.middleware.rate_limiter.settings", mock_settings):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_disabled_rate_limiting(self):
        app = FastAPI()
        with patch("proxy.middleware.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = False
            mock_settings.rate_limit_requests = 1
            mock_settings.rate_limit_window_seconds = 60
            mock_settings.redis_url = "redis://fake:6379/0"
            app.add_middleware(RateLimiterMiddleware)

            @app.get("/test")
            async def test_endpoint():
                return {"ok": True}

        client = TestClient(app)
        with patch("proxy.middleware.rate_limiter.settings", mock_settings):
            # Should allow all requests when disabled
            for _ in range(10):
                resp = client.get("/test")
                assert resp.status_code == 200

    async def test_check_rate_limit_fails_open(self):
        """When Redis is down, the rate limiter should fail open (allow requests)."""
        middleware = RateLimiterMiddleware.__new__(RateLimiterMiddleware)
        middleware.rate = 10
        middleware.window = 60
        middleware._redis = None
        middleware._script_sha = None

        with patch("proxy.middleware.rate_limiter.settings") as mock_settings:
            mock_settings.redis_url = "redis://nonexistent:6379/0"
            allowed, remaining = await middleware._check_rate_limit("test-key")
            assert allowed is True
            assert remaining == 10

    def test_rate_limit_headers_present(self):
        """Rate limit headers should be set on successful responses."""
        app = FastAPI()
        with patch("proxy.middleware.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window_seconds = 60
            mock_settings.redis_url = "redis://fake:6379/0"
            app.add_middleware(RateLimiterMiddleware, rate=100, window=60)

            @app.get("/test")
            async def test_endpoint():
                return {"ok": True}

        client = TestClient(app)
        with patch("proxy.middleware.rate_limiter.settings", mock_settings):
            # The first request should succeed (Redis will fail, so it fails open)
            resp = client.get("/test")
            assert resp.status_code == 200
            # When Redis is down, headers still get set with fallback values
            assert "X-RateLimit-Limit" in resp.headers


class TestSlidingWindowScript:
    def test_script_is_valid_lua(self):
        """Basic validation that the Lua script is non-empty and syntactically plausible."""
        assert "ZREMRANGEBYSCORE" in SLIDING_WINDOW_SCRIPT
        assert "ZADD" in SLIDING_WINDOW_SCRIPT
        assert "ZCARD" in SLIDING_WINDOW_SCRIPT
        assert "EXPIRE" in SLIDING_WINDOW_SCRIPT
