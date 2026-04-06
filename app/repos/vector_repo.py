"""
Repository layer for pgvector-backed vector storage.

Isolates all raw SQL behind a protocol so that the ``PgVectorIndex``
adapter never issues queries directly.

Production-safe for:
    FastAPI + SQLAlchemy 2.0 async + asyncpg + pgvector
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class VectorRepository(Protocol):
    """Persistence interface for document vectors."""

    async def upsert_batch(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunk_indexes: list[int],
        embeddings: list[list[float]],
    ) -> None:
        ...

    async def delete_by_document(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> int:
        ...


class PgVectorRepository:
    """
    pgvector-backed implementation of VectorRepository.

    Uses raw SQL via sqlalchemy.text() because the `vector`
    column type is not natively supported by SQLAlchemy ORM.

    Table:
        document_vectors
    """

    async def upsert_batch(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunk_indexes: list[int],
        embeddings: list[list[float]],
        chunk_texts: list[str] | None = None,
    ) -> None:
        """
        Upsert vectors for document chunks.

        Optimized: executemany (single prepared statement + many params)
        instead of N separate execute() calls.

        Uses CAST(:embedding AS vector) to avoid asyncpg colon parsing issues.
        """
        if not chunk_indexes or not embeddings:
            return

        if len(chunk_indexes) != len(embeddings):
            raise ValueError(
                f"chunk_indexes and embeddings length mismatch: "
                f"{len(chunk_indexes)} != {len(embeddings)}"
            )

        stmt = text(
            """
            INSERT INTO document_vectors
                (tenant_id, document_id, version_id, chunk_index, embedding, chunk_text)
            VALUES
                (:tenant_id, :document_id, :version_id, :chunk_index, CAST(:embedding AS vector), :chunk_text)
            ON CONFLICT (tenant_id, document_id, version_id, chunk_index)
            DO UPDATE SET embedding = EXCLUDED.embedding,
                          chunk_text = EXCLUDED.chunk_text
            """
        )

        texts = chunk_texts or [""] * len(chunk_indexes)

        rows: list[dict] = []
        for chunk_index, embedding, ctext in zip(chunk_indexes, embeddings, texts):
            # Convert Python list[float] → pgvector literal string: "[0.1,0.2,...]"
            embedding_literal = "[" + ",".join(str(v) for v in embedding) + "]"
            rows.append(
                {
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "version_id": version_id,
                    "chunk_index": int(chunk_index),
                    "embedding": embedding_literal,
                    "chunk_text": ctext,
                }
            )

        # ✅ executemany: SQLAlchemy will send as many parameter sets efficiently
        await db.execute(stmt, rows)

        logger.info(
            "vector_repo.upserted tenant_id=%s doc_id=%s version_id=%s rows=%d",
            tenant_id,
            document_id,
            version_id,
            len(rows),
        )

    async def delete_by_document(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> int:
        """
        Delete all vectors for a specific document version.
        """
        result = await db.execute(
            text(
                """
                DELETE FROM document_vectors
                WHERE tenant_id = :tenant_id
                  AND document_id = :document_id
                  AND version_id = :version_id
                """
            ),
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "version_id": version_id,
            },
        )

        deleted = result.rowcount or 0

        logger.info(
            "vector_repo.deleted tenant_id=%s doc_id=%s version_id=%s rows=%d",
            tenant_id,
            document_id,
            version_id,
            deleted,
        )

        return deleted