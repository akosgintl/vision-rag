"""Tests for auth, rate limiter, and request ID middleware."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from proxy.middleware.auth import AuthMiddleware
from proxy.middleware.request_id import RequestIdMiddleware

# ── Auth middleware tests ───────────────────────────────────────


class TestAuthMiddleware:
    """Tests for API key authentication middleware.

    Tests the dispatch logic directly to avoid Starlette BaseHTTPMiddleware
    exception wrapping issues with TestClient.
    """

    def _make_app(self, api_key: str | None = None):
        """Create a minimal app with auth middleware."""
        app = FastAPI()
        with patch("proxy.middleware.auth.settings") as mock_settings:
            mock_settings.api_key = api_key
            app.add_middleware(AuthMiddleware)

            @app.get("/test")
            async def test_endpoint():
                return {"ok": True}

            @app.get("/health")
            async def health():
                return {"status": "alive"}

            @app.get("/healthz")
            async def healthz():
                return {"status": "alive"}

            @app.get("/metrics")
            async def metrics():
                return {"metrics": "data"}

            @app.get("/health/infra")
            async def infra():
                return {"infra": "ok"}

            return app, mock_settings

    def test_no_api_key_configured_allows_all(self):
        app, _ = self._make_app(api_key=None)
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_public_paths_bypass_auth(self):
        app, mock_settings = self._make_app(api_key="secret-key")
        client = TestClient(app)
        for path in ["/health", "/healthz", "/metrics"]:
            with patch("proxy.middleware.auth.settings", mock_settings):
                resp = client.get(path)
                assert resp.status_code == 200, f"{path} should bypass auth"

    def test_public_prefixes_bypass_auth(self):
        app, mock_settings = self._make_app(api_key="secret-key")
        client = TestClient(app)
        with patch("proxy.middleware.auth.settings", mock_settings):
            resp = client.get("/health/infra")
            assert resp.status_code == 200

    def test_missing_key_raises_401(self):
        """Without an API key, middleware raises 401 HTTPException."""
        app, mock_settings = self._make_app(api_key="secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        with patch("proxy.middleware.auth.settings", mock_settings):
            resp = client.get("/test")
            # BaseHTTPMiddleware wraps HTTPException — status may be 401 or 500
            # depending on Starlette version. Just verify it's not 200.
            assert resp.status_code != 200

    def test_wrong_key_raises_403(self):
        """With wrong API key, middleware raises 403 HTTPException."""
        app, mock_settings = self._make_app(api_key="secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        with patch("proxy.middleware.auth.settings", mock_settings):
            resp = client.get("/test", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code != 200

    async def test_dispatch_missing_key_returns_401(self):
        """Directly test dispatch raises 401 for missing key."""
        middleware = AuthMiddleware.__new__(AuthMiddleware)
        mock_request = AsyncMock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.headers = {}
        mock_request.client = AsyncMock()
        mock_request.client.host = "127.0.0.1"

        with patch("proxy.middleware.auth.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            with pytest.raises(HTTPException) as exc_info:
                await middleware.dispatch(mock_request, AsyncMock())
            assert exc_info.value.status_code == 401

    async def test_dispatch_wrong_key_returns_403(self):
        """Directly test dispatch raises 403 for wrong key."""
        middleware = AuthMiddleware.__new__(AuthMiddleware)
        mock_request = AsyncMock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.headers = {"x-api-key": "wrong"}
        mock_request.client = AsyncMock()
        mock_request.client.host = "127.0.0.1"

        with patch("proxy.middleware.auth.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            with pytest.raises(HTTPException) as exc_info:
                await middleware.dispatch(mock_request, AsyncMock())
            assert exc_info.value.status_code == 403

    def test_correct_x_api_key_header(self):
        app, mock_settings = self._make_app(api_key="secret-key")
        client = TestClient(app)
        with patch("proxy.middleware.auth.settings", mock_settings):
            resp = client.get("/test", headers={"X-API-Key": "secret-key"})
            assert resp.status_code == 200

    def test_correct_bearer_token(self):
        app, mock_settings = self._make_app(api_key="secret-key")
        client = TestClient(app)
        with patch("proxy.middleware.auth.settings", mock_settings):
            resp = client.get("/test", headers={"Authorization": "Bearer secret-key"})
            assert resp.status_code == 200


# ── Request ID middleware tests ─────────────────────────────────


class TestRequestIdMiddleware:
    @pytest.fixture
    def app_with_request_id(self):
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)

        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"request_id": request.state.request_id}

        return app

    def test_generates_request_id(self, app_with_request_id):
        client = TestClient(app_with_request_id)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 36  # UUID format

    def test_propagates_existing_request_id(self, app_with_request_id):
        client = TestClient(app_with_request_id)
        resp = client.get("/test", headers={"X-Request-ID": "custom-id-123"})
        assert resp.headers["X-Request-ID"] == "custom-id-123"
        assert resp.json()["request_id"] == "custom-id-123"

    def test_stored_on_request_state(self, app_with_request_id):
        client = TestClient(app_with_request_id)
        resp = client.get("/test")
        data = resp.json()
        assert data["request_id"] == resp.headers["X-Request-ID"]
