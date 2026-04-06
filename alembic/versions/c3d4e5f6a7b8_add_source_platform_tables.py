"""add source platform tables

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-20 01:20:00.000000

Phase 4: Operationalize Source Platform.

Adds onboarded_sources and source_sync_runs tables for DB-backed
source management and sync tracking.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create onboarded_sources and source_sync_runs tables."""

    # ── onboarded_sources ────────────────────────────────────────
    op.create_table(
        'onboarded_sources',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('source_key', sa.String(length=128), nullable=False),
        sa.Column('name', sa.String(length=256), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('connector_type', sa.String(length=64), nullable=False),
        sa.Column('base_url', sa.String(length=1024), nullable=False),
        sa.Column('auth_type', sa.String(length=32), server_default=sa.text("'bearer'"), nullable=False),
        sa.Column('auth_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('list_path', sa.String(length=512), server_default=sa.text("'/api/internal/knowledge/items'"), nullable=False),
        sa.Column('detail_path_template', sa.String(length=512), nullable=True),
        sa.Column('request_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('mapping_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('default_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('last_synced_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], name='fk_onboarded_sources_tenant_id'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'source_key', name='uq_onboarded_sources_tenant_source_key'),
    )
    op.create_index('idx_onboarded_sources_tenant_id', 'onboarded_sources', ['tenant_id'])
    op.create_index('idx_onboarded_sources_connector_type', 'onboarded_sources', ['connector_type'])
    op.create_index('idx_onboarded_sources_is_active', 'onboarded_sources', ['is_active'])
    op.create_index('idx_onboarded_sources_tenant_active', 'onboarded_sources', ['tenant_id', 'is_active'])

    # ── source_sync_runs ─────────────────────────────────────────
    op.create_table(
        'source_sync_runs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('source_id', sa.BigInteger(), nullable=False),
        sa.Column('source_key', sa.String(length=128), nullable=False),
        sa.Column('status', sa.String(length=32), server_default=sa.text("'running'"), nullable=False),
        sa.Column('started_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('finished_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('items_fetched', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('items_upserted', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('items_failed', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('triggered_by', sa.String(length=32), nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['onboarded_sources.id'], name='fk_sync_runs_source_id', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_sync_runs_tenant_id', 'source_sync_runs', ['tenant_id'])
    op.create_index('idx_sync_runs_source_id', 'source_sync_runs', ['source_id'])
    op.create_index('idx_sync_runs_status', 'source_sync_runs', ['status'])
    op.create_index('idx_sync_runs_started_at', 'source_sync_runs', ['started_at'])
    op.create_index('idx_sync_runs_tenant_source', 'source_sync_runs', ['tenant_id', 'source_id'])


def downgrade() -> None:
    """Drop source_sync_runs and onboarded_sources tables."""
    # Drop sync_runs first (FK dependency)
    op.drop_index('idx_sync_runs_tenant_source', table_name='source_sync_runs')
    op.drop_index('idx_sync_runs_started_at', table_name='source_sync_runs')
    op.drop_index('idx_sync_runs_status', table_name='source_sync_runs')
    op.drop_index('idx_sync_runs_source_id', table_name='source_sync_runs')
    op.drop_index('idx_sync_runs_tenant_id', table_name='source_sync_runs')
    op.drop_table('source_sync_runs')

    op.drop_index('idx_onboarded_sources_tenant_active', table_name='onboarded_sources')
    op.drop_index('idx_onboarded_sources_is_active', table_name='onboarded_sources')
    op.drop_index('idx_onboarded_sources_connector_type', table_name='onboarded_sources')
    op.drop_index('idx_onboarded_sources_tenant_id', table_name='onboarded_sources')
    op.drop_table('onboarded_sources')
