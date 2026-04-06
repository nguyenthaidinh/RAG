"""add source scheduling and health columns

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-20 11:10:00.000000

Phase 7: Scheduled Sync + Source Reliability Hardening.

Adds scheduling fields (sync_enabled, sync_interval_minutes, next_sync_at,
last_sync_attempt_at) and health tracking fields (last_success_at,
last_failure_at, consecutive_failures, last_error_message) to
onboarded_sources.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add scheduling and health columns to onboarded_sources."""

    # ── Scheduling fields ────────────────────────────────────────
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'sync_enabled', sa.Boolean(),
            server_default=sa.text('false'), nullable=False,
        ),
    )
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'sync_interval_minutes', sa.Integer(),
            server_default=sa.text('60'), nullable=False,
        ),
    )
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'last_sync_attempt_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'next_sync_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )

    # ── Health tracking fields ───────────────────────────────────
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'last_success_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'last_failure_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'consecutive_failures', sa.Integer(),
            server_default=sa.text('0'), nullable=False,
        ),
    )
    op.add_column(
        'onboarded_sources',
        sa.Column(
            'last_error_message', sa.Text(),
            nullable=True,
        ),
    )

    # ── Index for scheduler due-source query ─────────────────────
    op.create_index(
        'idx_onboarded_sources_next_sync',
        'onboarded_sources',
        ['sync_enabled', 'is_active', 'next_sync_at'],
    )


def downgrade() -> None:
    """Remove scheduling and health columns."""
    op.drop_index(
        'idx_onboarded_sources_next_sync',
        table_name='onboarded_sources',
    )
    op.drop_column('onboarded_sources', 'last_error_message')
    op.drop_column('onboarded_sources', 'consecutive_failures')
    op.drop_column('onboarded_sources', 'last_failure_at')
    op.drop_column('onboarded_sources', 'last_success_at')
    op.drop_column('onboarded_sources', 'next_sync_at')
    op.drop_column('onboarded_sources', 'last_sync_attempt_at')
    op.drop_column('onboarded_sources', 'sync_interval_minutes')
    op.drop_column('onboarded_sources', 'sync_enabled')
