"""
Request ID middleware — generates/propagates X-Request-ID for distributed tracing.

Every incoming request gets a unique ID that is:
1. Read from the X-Request-ID header (if present, e.g. from a load balancer)
2. Or auto-generated as a UUID
3. Bound to structlog context for the duration of the request
4. Added to the response X-Request-ID header
5. Stored in request.state.request_id for use in route handlers
"""

import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Injects and propagates X-Request-ID across request lifecycle."""

    async def dispatch(self, request: Request, call_next):
        # Use existing header or generate new ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind request_id to structlog context for all log calls during this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
