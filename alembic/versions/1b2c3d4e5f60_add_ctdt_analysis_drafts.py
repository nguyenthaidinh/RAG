"""add ctdt_analysis_drafts table

Revision ID: 1b2c3d4e5f60
Revises: 0a1b2c3d4e5f
Create Date: 2026-05-17 00:00:00.000000

R5.5: Persist CTDT update-cycle analysis drafts separately from official
Program / ProgramVersion / ProgramVersionRevision data.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "1b2c3d4e5f60"
down_revision: Union[str, Sequence[str], None] = "0a1b2c3d4e5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create ctdt_analysis_drafts table."""
    op.create_table(
        "ctdt_analysis_drafts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("update_cycle_id", sa.String(length=64), nullable=False),
        sa.Column("program_id", sa.String(length=64), nullable=True),
        sa.Column("program_code", sa.String(length=64), nullable=True),
        sa.Column("program_name", sa.String(length=256), nullable=True),
        sa.Column("analysis_mode", sa.String(length=32), nullable=False),
        sa.Column(
            "draft_type",
            sa.String(length=64),
            server_default=sa.text("'update_cycle_analysis'"),
            nullable=False,
        ),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by", sa.BigInteger(), nullable=True),
        sa.Column("updated_by", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'draft'"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'archived')",
            name="ck_ctdt_analysis_drafts_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_ctdt_analysis_drafts_tenant_cycle",
        "ctdt_analysis_drafts",
        ["tenant_id", "update_cycle_id"],
    )
    op.create_index(
        "idx_ctdt_analysis_drafts_tenant_cycle_program",
        "ctdt_analysis_drafts",
        ["tenant_id", "update_cycle_id", "program_code"],
    )
    op.create_index(
        "idx_ctdt_analysis_drafts_tenant_type",
        "ctdt_analysis_drafts",
        ["tenant_id", "draft_type"],
    )
    op.create_index(
        "idx_ctdt_analysis_drafts_latest",
        "ctdt_analysis_drafts",
        ["tenant_id", "update_cycle_id", "analysis_mode", "updated_at"],
    )


def downgrade() -> None:
    """Drop ctdt_analysis_drafts table."""
    op.drop_index("idx_ctdt_analysis_drafts_latest", table_name="ctdt_analysis_drafts")
    op.drop_index("idx_ctdt_analysis_drafts_tenant_type", table_name="ctdt_analysis_drafts")
    op.drop_index("idx_ctdt_analysis_drafts_tenant_cycle_program", table_name="ctdt_analysis_drafts")
    op.drop_index("idx_ctdt_analysis_drafts_tenant_cycle", table_name="ctdt_analysis_drafts")
    op.drop_table("ctdt_analysis_drafts")
