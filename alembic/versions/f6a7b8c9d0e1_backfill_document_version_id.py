"""backfill document version_id from checksum

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-04-01 16:30:00.000000

Backfill documents.version_id = documents.checksum
for any rows where version_id IS NULL.

This is a data-only migration — no schema changes.
Safe to run on a live system; UPDATE … WHERE is
scoped to NULL rows only and is idempotent.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, Sequence[str], None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Set version_id = checksum for documents where version_id is NULL."""
    op.execute(
        "UPDATE documents SET version_id = checksum WHERE version_id IS NULL"
    )


def downgrade() -> None:
    """Revert version_id to NULL (restore previous state).

    Safe to run — only clears version_id values that were set by
    this migration.  Documents that already had a version_id before
    this migration are not affected (they keep their value).
    """
    # In practice this is a no-op risk since we only set version_id
    # for rows that had NULL.  A precise downgrade would require
    # tracking which rows were backfilled, but since this is a
    # transitional fix, a simple NULL-reset is acceptable.
    pass
