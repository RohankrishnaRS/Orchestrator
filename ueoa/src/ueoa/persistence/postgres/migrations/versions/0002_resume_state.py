"""Add run, workflow, approval, and audit fields required by current models.

Revision ID: 0002_resume_state
Revises: 0001_initial
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, Text, text
from sqlalchemy.dialects.postgresql import JSONB

SCHEMA = "ueoa"

revision = "0002_resume_state"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workflows",
        Column("extra_data", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        schema=SCHEMA,
    )
    op.add_column(
        "runs",
        Column("workflow", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        schema=SCHEMA,
    )
    op.add_column("runs", Column("error", Text), schema=SCHEMA)
    op.add_column("run_nodes", Column("item_data", JSONB), schema=SCHEMA)
    op.add_column(
        "approvals",
        Column("item_key", Text, nullable=False, server_default=text("''")),
        schema=SCHEMA,
    )
    op.add_column(
        "approvals",
        Column("payload", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        schema=SCHEMA,
    )
    op.add_column(
        "approvals",
        Column("idempotency_key", Text, nullable=False, server_default=text("''")),
        schema=SCHEMA,
    )
    op.add_column("audit_log", Column("output", JSONB), schema=SCHEMA)


def downgrade() -> None:
    op.drop_column("approvals", "idempotency_key", schema=SCHEMA)
    op.drop_column("approvals", "payload", schema=SCHEMA)
    op.drop_column("approvals", "item_key", schema=SCHEMA)
    op.drop_column("audit_log", "output", schema=SCHEMA)
    op.drop_column("run_nodes", "item_data", schema=SCHEMA)
    op.drop_column("runs", "error", schema=SCHEMA)
    op.drop_column("runs", "workflow", schema=SCHEMA)
    op.drop_column("workflows", "extra_data", schema=SCHEMA)
