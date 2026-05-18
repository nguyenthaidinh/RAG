import asyncio
from sqlalchemy import text

from app.db.session import engine
from app.db.models import Base


async def main():
    async with engine.begin() as conn:
        # Tạo toàn bộ bảng từ ORM models hiện tại
        await conn.run_sync(Base.metadata.create_all)

        # Alembic is the source of truth for production schema. This script
        # mirrors key Alembic DDL only for local/dev bootstrap convenience.

        # Bật pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Bảng vector chưa có ORM model riêng, tạo thủ công
        await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS document_vectors (
            id BIGSERIAL PRIMARY KEY,
            tenant_id VARCHAR(64) NOT NULL,
            document_id BIGINT NOT NULL,
            version_id VARCHAR(64) NOT NULL,
            chunk_index INT NOT NULL,
            embedding vector NOT NULL,
            chunk_text TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_docvec_tenant_doc_version_chunk
                UNIQUE (tenant_id, document_id, version_id, chunk_index)
        )
        """))

        await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_docvec_tenant_doc
        ON document_vectors (tenant_id, document_id)
        """))

        await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_docvec_tenant_doc_version
        ON document_vectors (tenant_id, document_id, version_id)
        """))

        # BM25 / FTS index cho documents
        await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_documents_content_fts_simple
        ON documents
        USING GIN (to_tsvector('simple', COALESCE(content_text, '')))
        """))

    print("✅ Local DB bootstrap completed.")


if __name__ == "__main__":
    asyncio.run(main())
