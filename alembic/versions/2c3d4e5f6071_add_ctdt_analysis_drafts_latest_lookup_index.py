"""add latest lookup index for ctdt analysis drafts

Revision ID: 2c3d4e5f6071
Revises: 1b2c3d4e5f60
Create Date: 2026-05-17 00:00:00.000000

R5.5 hardening: optimize latest active draft lookup without modifying the
existing ctdt_analysis_drafts migration.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2c3d4e5f6071"
down_revision: Union[str, Sequence[str], None] = "1b2c3d4e5f60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create optimized latest lookup index."""
    op.execute(sa.text(
        """
        CREATE INDEX IF NOT EXISTS idx_ctdt_analysis_drafts_latest_lookup
        ON ctdt_analysis_drafts (
            tenant_id,
            update_cycle_id,
            program_code,
            analysis_mode,
            draft_type,
            updated_at DESC,
            id DESC
        )
        """
    ))


def downgrade() -> None:
    """Drop optimized latest lookup index."""
    op.execute(sa.text(
        "DROP INDEX IF EXISTS idx_ctdt_analysis_drafts_latest_lookup"
    ))
