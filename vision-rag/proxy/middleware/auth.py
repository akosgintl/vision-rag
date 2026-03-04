"""API key authentication middleware."""

import hmac

import structlog
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from proxy.config import settings

logger = structlog.get_logger()

# Paths that don't require authentication
PUBLIC_PATHS = {"/health", "/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"}
PUBLIC_PREFIXES = ("/health/",)  # /health/{backend}, /health/circuits, /health/infra


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no API key is configured
        if not settings.api_key:
            return await call_next(request)

        # Skip public paths
        path = request.url.path
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("x-api-key") or request.headers.get("authorization", "").replace("Bearer ", "")

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide via 'X-API-Key' header or 'Authorization: Bearer <key>'.",
            )

        if not hmac.compare_digest(api_key.encode(), settings.api_key.encode()):
            logger.warning("auth_failed", client_ip=request.client.host if request.client else "unknown")
            raise HTTPException(
                status_code=403,
                detail="Invalid API key.",
            )

        return await call_next(request)
