"""
PostgreSQL document metadata service.

Tracks ingested documents, their status, and page counts.
Uses psycopg2 with connection pooling via asyncio.run_in_executor
to avoid blocking the event loop.
"""

import asyncio
from functools import partial

import psycopg2
import structlog
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from proxy.config import settings

logger = structlog.get_logger()

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    collection      TEXT NOT NULL DEFAULT 'default',
    filename        TEXT,
    source_url      TEXT,
    page_count      INTEGER DEFAULT 0,
    indexed_pages   INTEGER DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    dpi             INTEGER DEFAULT 300,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""


class DocumentMetadata:
    """PostgreSQL-backed document metadata store with connection pooling."""

    def __init__(self, min_conn: int = 2, max_conn: int = 10):
        self.pool = pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            dsn=settings.postgres_dsn,
        )

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def ensure_tables(self) -> None:
        """Create tables if they don't exist. Call once at startup."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()
            logger.info("postgres_tables_ready")
        finally:
            self._put_conn(conn)

    def _register_document_sync(
        self,
        document_id: str,
        collection: str,
        filename: str | None = None,
        source_url: str | None = None,
        dpi: int = 300,
        metadata: dict | None = None,
    ) -> dict:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO documents (document_id, collection, filename, source_url, dpi, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (document_id) DO UPDATE SET
                        collection = EXCLUDED.collection,
                        filename = EXCLUDED.filename,
                        source_url = EXCLUDED.source_url,
                        dpi = EXCLUDED.dpi,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING *
                    """,
                    (
                        document_id,
                        collection,
                        filename,
                        source_url,
                        dpi,
                        psycopg2.extras.Json(metadata or {}),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            return dict(row)
        finally:
            self._put_conn(conn)

    async def register_document(self, **kwargs) -> dict:
        """Register a new document (async wrapper)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._register_document_sync, **kwargs))

    def _update_document_sync(
        self,
        document_id: str,
        page_count: int | None = None,
        indexed_pages: int | None = None,
        status: str | None = None,
    ) -> dict | None:
        conn = self._get_conn()
        try:
            sets = []
            params = []
            if page_count is not None:
                sets.append("page_count = %s")
                params.append(page_count)
            if indexed_pages is not None:
                sets.append("indexed_pages = %s")
                params.append(indexed_pages)
            if status is not None:
                sets.append("status = %s")
                params.append(status)

            if not sets:
                return None

            sets.append("updated_at = NOW()")
            params.append(document_id)

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"UPDATE documents SET {', '.join(sets)} WHERE document_id = %s RETURNING *",
                    params,
                )
                row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None
        finally:
            self._put_conn(conn)

    async def update_document(self, **kwargs) -> dict | None:
        """Update document metadata (async wrapper)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._update_document_sync, **kwargs))

    def _get_document_sync(self, document_id: str) -> dict | None:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE document_id = %s", (document_id,))
                row = cur.fetchone()
            return dict(row) if row else None
        finally:
            self._put_conn(conn)

    async def get_document(self, document_id: str) -> dict | None:
        """Get document by ID (async wrapper)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._get_document_sync, document_id))

    def _list_documents_sync(self, collection: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if collection:
                    cur.execute(
                        "SELECT * FROM documents WHERE collection = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (collection, limit, offset),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                return [dict(row) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)

    async def list_documents(self, collection: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        """List documents with optional collection filter (async wrapper)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._list_documents_sync, collection, limit, offset))

    def close(self) -> None:
        """Close the connection pool."""
        self.pool.closeall()
        logger.info("postgres_pool_closed")
