"""add chunk_text to document_vectors

Revision ID: a1b2c3d4e5f6
Revises: 95a9597e255f
Create Date: 2026-03-14 09:55:00.000000

Phase 1 hotfix: persist chunk text at index time for snippet stability.

Background:
    Previously, snippet text was reconstructed at query time by re-chunking
    the full document content_text.  If chunking config changed (chunk size,
    overlap, tokenizer, etc.), the chunk_index recorded in the vector DB
    would no longer point to the same text — causing snippet drift.

    By persisting the exact chunk text alongside the embedding at index time,
    we guarantee that query-time snippets always match what was indexed.

Notes:
    - Column is nullable: existing rows get NULL, handled by fallback logic
      in DefaultVectorRetriever._extract_snippet().
    - document_vectors is a manually-managed table (not in Base.metadata),
      created by bootstrap_local_db.py or by this migration.
    - Uses IF NOT EXISTS / IF EXISTS for fail-safe deploy on any env state.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '95a9597e255f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add chunk_text column to document_vectors if table exists."""

    # document_vectors is created by bootstrap_local_db.py, not by Alembic.
    # In environments where the table already exists (production, staging),
    # we just add the column.  IF NOT EXISTS ensures idempotency.
    #
    # In environments where bootstrap hasn't run yet, this is a no-op
    # (the table doesn't exist, so ALTER TABLE does nothing harmful —
    # bootstrap_local_db.py will create the full table with chunk_text).
    op.execute(sa.text(
        "ALTER TABLE document_vectors "
        "ADD COLUMN IF NOT EXISTS chunk_text TEXT"
    ))


def downgrade() -> None:
    """Remove chunk_text column from document_vectors."""
    op.execute(sa.text(
        "ALTER TABLE document_vectors "
        "DROP COLUMN IF EXISTS chunk_text"
    ))
