"""add document_events table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-15 19:48:00.000000

Phase 2A: Document Admin Foundation.

Adds the document_events table for append-only document lifecycle
history tracking. This is separate from audit_events — it stores
structured, domain-specific events for document admin views.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document_events table."""
    op.create_table(
        'document_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('document_id', sa.BigInteger(), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('from_status', sa.String(length=50), nullable=True),
        sa.Column('to_status', sa.String(length=50), nullable=True),
        sa.Column('actor_user_id', sa.BigInteger(), nullable=True),
        sa.Column('request_id', sa.String(length=128), nullable=True),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_document_events_tenant_id', 'document_events', ['tenant_id'])
    op.create_index('ix_document_events_document_id', 'document_events', ['document_id'])
    op.create_index('ix_document_events_tenant_document', 'document_events', ['tenant_id', 'document_id'])
    op.create_index('ix_document_events_event_type', 'document_events', ['event_type'])
    op.create_index('ix_document_events_created_at', 'document_events', ['created_at'])


def downgrade() -> None:
    """Drop document_events table."""
    op.drop_index('ix_document_events_created_at', table_name='document_events')
    op.drop_index('ix_document_events_event_type', table_name='document_events')
    op.drop_index('ix_document_events_tenant_document', table_name='document_events')
    op.drop_index('ix_document_events_document_id', table_name='document_events')
    op.drop_index('ix_document_events_tenant_id', table_name='document_events')
    op.drop_table('document_events')
