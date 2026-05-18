"""create document_vectors and postgres fts indexes

Revision ID: 0a1b2c3d4e5f
Revises: f6a7b8c9d0e1
Create Date: 2026-05-16 00:00:00.000000

Alembic is the source of truth for document_vectors. The statements are
idempotent so environments that already created the table via local bootstrap
can upgrade without destructive changes.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0a1b2c3d4e5f"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create pgvector storage and FTS indexes idempotently."""
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    op.execute(sa.text(
        """
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
        """
    ))

    op.execute(sa.text(
        """
        ALTER TABLE document_vectors
        ADD COLUMN IF NOT EXISTS chunk_text TEXT
        """
    ))

    # Existing tables may have been created before Alembic owned this table.
    # A unique index is enough for ON CONFLICT (tenant_id, document_id,
    # version_id, chunk_index) even if the named table constraint pre-existed
    # or was missing.
    op.execute(sa.text(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_docvec_tenant_doc_version_chunk
        ON document_vectors (tenant_id, document_id, version_id, chunk_index)
        """
    ))

    op.execute(sa.text(
        """
        CREATE INDEX IF NOT EXISTS idx_docvec_tenant_doc
        ON document_vectors (tenant_id, document_id)
        """
    ))

    op.execute(sa.text(
        """
        CREATE INDEX IF NOT EXISTS idx_docvec_tenant_doc_version
        ON document_vectors (tenant_id, document_id, version_id)
        """
    ))

    op.execute(sa.text(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_content_fts_simple
        ON documents
        USING GIN (to_tsvector('simple', COALESCE(content_text, '')))
        """
    ))


def downgrade() -> None:
    """Conservative downgrade: keep vector data, remove only added indexes."""
    op.execute(sa.text("DROP INDEX IF EXISTS idx_documents_content_fts_simple"))
    op.execute(sa.text("DROP INDEX IF EXISTS idx_docvec_tenant_doc_version"))
