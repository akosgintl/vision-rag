"""Redis-backed sliding window rate limiter middleware."""

import time

import redis.asyncio as aioredis
import structlog
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from proxy.config import settings

logger = structlog.get_logger()

# Lua script for atomic sliding-window rate limiting.
# Returns [is_allowed (0 or 1), remaining_count].
SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cutoff = now - window

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now .. ':' .. math.random(1, 1000000))
    redis.call('EXPIRE', key, window + 1)
    return {1, limit - count - 1}
else
    redis.call('EXPIRE', key, window + 1)
    return {0, 0}
end
"""


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting by API key or IP.

    Uses Redis sorted sets for a sliding window counter that works
    correctly across multiple workers and pods.
    Falls back to allowing requests if Redis is unavailable.
    """

    def __init__(self, app, rate: int | None = None, window: int | None = None):
        super().__init__(app)
        self.rate = rate or settings.rate_limit_requests
        self.window = window or settings.rate_limit_window_seconds
        self._redis: aioredis.Redis | None = None
        self._script_sha: str | None = None

    async def _get_redis(self) -> aioredis.Redis:
        """Lazy-init Redis connection."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            # Pre-load the Lua script
            self._script_sha = await self._redis.script_load(SLIDING_WINDOW_SCRIPT)
        return self._redis

    async def _check_rate_limit(self, key: str) -> tuple[bool, int]:
        """
        Check and consume a rate limit token.
        Returns (is_allowed, remaining).
        """
        try:
            redis = await self._get_redis()
            now = time.time()
            result = await redis.evalsha(
                self._script_sha,
                1,
                f"vrag:ratelimit:{key}",
                str(self.window),
                str(self.rate),
                str(now),
            )
            return bool(result[0]), int(result[1])
        except Exception as e:
            # Redis down — fail open (allow the request)
            logger.warning("rate_limiter_redis_error", error=str(e))
            self._redis = None
            self._script_sha = None
            return True, self.rate

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Skip health and metrics endpoints
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        # Use API key or client IP as the rate limit key
        key = (
            request.headers.get("x-api-key")
            or request.headers.get("authorization", "").replace("Bearer ", "")
            or (request.client.host if request.client else "unknown")
        )

        allowed, remaining = await self._check_rate_limit(key)

        if not allowed:
            logger.warning("rate_limited", key_hash=hash(key) % 10000, remaining=remaining)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": self.rate,
                    "window_seconds": self.window,
                    "retry_after": self.window,
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.window)

        return response
