"""add source document links table and sync run metric columns

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-25 01:00:00.000000

Phase 8: Delta-Aware Sync.

Adds source_document_links table for source item ↔ document mapping,
and enriches source_sync_runs with granular metric columns.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create source_document_links and add metric columns to source_sync_runs."""

    # ── source_document_links ────────────────────────────────────
    op.create_table(
        'source_document_links',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('onboarded_source_id', sa.BigInteger(), nullable=False),
        sa.Column('source_key', sa.String(length=128), nullable=False),
        sa.Column('external_id', sa.String(length=512), nullable=False),
        sa.Column('external_uri', sa.String(length=1024), nullable=True),
        sa.Column('document_id', sa.BigInteger(), nullable=True),
        sa.Column('document_version_id', sa.Text(), nullable=True),
        sa.Column('content_checksum', sa.String(length=64), nullable=True),
        sa.Column(
            'remote_updated_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'last_seen_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'last_synced_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'status', sa.String(length=32),
            server_default=sa.text("'active'"), nullable=False,
        ),
        sa.Column(
            'metadata_json',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['tenant_id'], ['tenants.id'],
            name='fk_source_doc_links_tenant_id',
        ),
        sa.ForeignKeyConstraint(
            ['onboarded_source_id'], ['onboarded_sources.id'],
            name='fk_source_doc_links_source_id',
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'tenant_id', 'onboarded_source_id', 'external_id',
            name='uq_source_doc_links_tenant_source_ext',
        ),
    )
    op.create_index(
        'idx_source_doc_links_tenant_source',
        'source_document_links',
        ['tenant_id', 'onboarded_source_id'],
    )
    op.create_index(
        'idx_source_doc_links_tenant_source_status',
        'source_document_links',
        ['tenant_id', 'onboarded_source_id', 'status'],
    )
    op.create_index(
        'idx_source_doc_links_document_id',
        'source_document_links',
        ['document_id'],
    )

    # ── Enrich source_sync_runs with granular metrics ────────────
    op.add_column(
        'source_sync_runs',
        sa.Column(
            'items_created', sa.Integer(),
            server_default=sa.text('0'), nullable=False,
        ),
    )
    op.add_column(
        'source_sync_runs',
        sa.Column(
            'items_updated', sa.Integer(),
            server_default=sa.text('0'), nullable=False,
        ),
    )
    op.add_column(
        'source_sync_runs',
        sa.Column(
            'items_unchanged', sa.Integer(),
            server_default=sa.text('0'), nullable=False,
        ),
    )
    op.add_column(
        'source_sync_runs',
        sa.Column(
            'items_missing', sa.Integer(),
            server_default=sa.text('0'), nullable=False,
        ),
    )
    op.add_column(
        'source_sync_runs',
        sa.Column(
            'items_reactivated', sa.Integer(),
            server_default=sa.text('0'), nullable=False,
        ),
    )


def downgrade() -> None:
    """Drop source_document_links and remove metric columns."""
    op.drop_column('source_sync_runs', 'items_reactivated')
    op.drop_column('source_sync_runs', 'items_missing')
    op.drop_column('source_sync_runs', 'items_unchanged')
    op.drop_column('source_sync_runs', 'items_updated')
    op.drop_column('source_sync_runs', 'items_created')

    op.drop_index(
        'idx_source_doc_links_document_id',
        table_name='source_document_links',
    )
    op.drop_index(
        'idx_source_doc_links_tenant_source_status',
        table_name='source_document_links',
    )
    op.drop_index(
        'idx_source_doc_links_tenant_source',
        table_name='source_document_links',
    )
    op.drop_table('source_document_links')
