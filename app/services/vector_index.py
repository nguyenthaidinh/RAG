# app/services/vector_index.py
"""
Vendor-agnostic vector index abstraction layer.

Protocol + adapters:

* **NullIndex** — no-op (default, no embedding needed).
* **FaissIndex** — in-process FAISS; NOT persisted, NOT production.
* **QdrantIndex** — Qdrant vector DB via ``qdrant-client``.
* **PgVectorIndex** — PostgreSQL pgvector via SQLAlchemy async.

PATCH: add PgVectorIndex.{upsert_in_tx, delete_in_tx} so DocumentService can
write vectors inside the caller transaction (no nested session/commit).
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.core.config import settings

if TYPE_CHECKING:
    from app.nlp.types import Chunk
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    """
    Pluggable vector-storage interface.

    Implementations receive pre-computed embeddings and are responsible
    only for storage and retrieval — never for generation.
    """

    async def upsert(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        ...

    async def delete(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> None:
        ...


class NullIndex:
    """No-op adapter — all methods succeed silently."""

    __slots__ = ()

    async def upsert(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        return

    async def delete(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> None:
        return


class FaissIndex:
    """
    In-process FAISS index.

    NOT persisted — data lives only in the current process.
    Explicitly not suitable for production.

    Requires: faiss-cpu, numpy.
    """

    def __init__(self, dim: int = 128) -> None:
        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FaissIndex requires 'faiss-cpu' and 'numpy'. "
                "Install them with: pip install faiss-cpu numpy"
            ) from exc

        flat = faiss.IndexFlatL2(dim)
        self._index = faiss.IndexIDMap(flat)
        self._dim = dim
        self._doc_ids: dict[tuple[str, int, str], list[int]] = {}
        self._next_id: int = 0

    async def upsert(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        import numpy as np  # type: ignore[import-untyped]

        await self.delete(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
        )

        vectors = np.array(embeddings, dtype=np.float32)
        ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
        self._index.add_with_ids(vectors, ids)

        key = (tenant_id, document_id, version_id)
        self._doc_ids[key] = ids.tolist()
        self._next_id += len(embeddings)

        logger.info(
            "faiss.upserted tenant_id=%s doc_id=%s version_id=%s vectors=%d",
            tenant_id, document_id, version_id, len(embeddings),
        )

    async def delete(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> None:
        import numpy as np  # type: ignore[import-untyped]

        key = (tenant_id, document_id, version_id)
        if key in self._doc_ids:
            ids = np.array(self._doc_ids[key], dtype=np.int64)
            self._index.remove_ids(ids)
            del self._doc_ids[key]
            logger.info(
                "faiss.deleted tenant_id=%s doc_id=%s version_id=%s ids=%d",
                tenant_id, document_id, version_id, len(ids),
            )


class QdrantIndex:
    """
    Qdrant-backed vector index.

    Creates collection if missing. Uses deterministic UUID5 point IDs
    derived from (tenant_id, document_id, version_id, chunk_index)
    for idempotent upserts.

    Requires: qdrant-client.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "QdrantIndex requires 'qdrant-client'. Install with: pip install qdrant-client"
            ) from exc

        self._url = url or settings.QDRANT_URL
        self._api_key = (api_key or settings.QDRANT_API_KEY) or None
        self._collection = collection or settings.QDRANT_COLLECTION
        self._dim = dim or settings.EMBEDDING_DIM
        self._client: AsyncQdrantClient = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        self._collection_ready = False

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]

        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info("qdrant.collection_created name=%s dim=%d", self._collection, self._dim)
        self._collection_ready = True

    @staticmethod
    def _point_id(tenant_id: str, document_id: int, version_id: str, chunk_index: int) -> str:
        key = f"{tenant_id}:{document_id}:{version_id}:{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    async def upsert(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

        await self._ensure_collection()

        points = [
            PointStruct(
                id=self._point_id(tenant_id, document_id, version_id, chunk.chunk_index),
                vector=emb,
                payload={
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "version_id": version_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.text,  # Phase 1: persist for snippet stability
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

        logger.info(
            "qdrant.upserted tenant_id=%s doc_id=%s version_id=%s points=%d",
            tenant_id, document_id, version_id, len(points),
        )

    async def delete(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import-untyped]

        await self._ensure_collection()

        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                ],
            ),
        )

        logger.info("qdrant.deleted tenant_id=%s doc_id=%s version_id=%s", tenant_id, document_id, version_id)


class PgVectorIndex:
    """
    pgvector-backed vector index using SQLAlchemy async.

    PATCH: provide *in-transaction* methods so caller can keep a single
    transaction boundary (no nested session/commit).
    """

    def __init__(self, session_factory=None) -> None:
        from app.repos.vector_repo import PgVectorRepository

        if session_factory is None:
            from app.db.session import AsyncSessionLocal
            session_factory = AsyncSessionLocal

        self._session_factory = session_factory
        self._repo = PgVectorRepository()

    # Legacy behavior (kept for compatibility): opens its own session/tx.
    async def upsert(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        async with self._session_factory() as db:
            async with db.begin():
                await self._repo.upsert_batch(
                    db,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version_id=version_id,
                    chunk_indexes=[c.chunk_index for c in chunks],
                    embeddings=embeddings,
                    chunk_texts=[c.text for c in chunks],
                )
        logger.info(
            "pgvector.upserted tenant_id=%s doc_id=%s version_id=%s vectors=%d",
            tenant_id, document_id, version_id, len(embeddings),
        )

    async def delete(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> None:
        async with self._session_factory() as db:
            async with db.begin():
                await self._repo.delete_by_document(
                    db,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version_id=version_id,
                )
        logger.info(
            "pgvector.deleted tenant_id=%s doc_id=%s version_id=%s",
            tenant_id, document_id, version_id,
        )

    # NEW: in-transaction methods for DocumentService (preferred)
    async def upsert_in_tx(
        self,
        db: "AsyncSession",
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        await self._repo.upsert_batch(
            db,
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            chunk_indexes=[c.chunk_index for c in chunks],
            embeddings=embeddings,
            chunk_texts=[c.text for c in chunks],
        )
        logger.info(
            "pgvector.upserted_in_tx tenant_id=%s doc_id=%s version_id=%s vectors=%d",
            tenant_id, document_id, version_id, len(embeddings),
        )

    async def delete_in_tx(
        self,
        db: "AsyncSession",
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> None:
        await self._repo.delete_by_document(
            db,
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
        )
        logger.info(
            "pgvector.deleted_in_tx tenant_id=%s doc_id=%s version_id=%s",
            tenant_id, document_id, version_id,
        )


def get_vector_index(index_type: str | None = None) -> VectorIndex:
    """
    Return a VectorIndex based on config or override.
    Falls back to NullIndex for unknown types.
    """
    name = (index_type or settings.VECTOR_INDEX).lower().strip()

    if name == "null":
        return NullIndex()
    if name == "faiss":
        return FaissIndex(dim=settings.EMBEDDING_DIM)
    if name == "qdrant":
        return QdrantIndex()
    if name == "pgvector":
        return PgVectorIndex()

    logger.warning("vector_index.factory unknown type=%s, falling back to null", name)
    return NullIndex()