"""Initial persistence schema, frozen at commit 31a7c7a.

Revision ID: 0001_initial
Revises:
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

SCHEMA = "ueoa"


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "capabilities",
        Column("id", Text, primary_key=True),
        Column("version", Text, primary_key=True),
        Column("major", Integer, nullable=False),
        Column("effects", Text, nullable=False),
        Column("contract", JSONB, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        schema=SCHEMA,
    )
    op.create_table(
        "implementations",
        Column("key", Text, primary_key=True),
        Column("agent_id", Text, nullable=False),
        Column("version", Text, nullable=False),
        Column("binding", Text, nullable=False),
        Column("kind", Text, nullable=False),
        Column("status", Text, nullable=False),
        Column("priority", Integer, nullable=False),
        Column("source", Text, nullable=False),
        Column("card", JSONB, nullable=False),
        Column("card_sha256", Text, nullable=False),
        Column("implements", ARRAY(Text), nullable=False),
        Column("checks", JSONB, nullable=False),
        Column("registered_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_implementations_implements",
        "implementations",
        ["implements"],
        schema=SCHEMA,
        postgresql_using="gin",
    )
    op.create_index("ix_implementations_status", "implementations", ["status"], schema=SCHEMA)
    op.create_table(
        "workflows",
        Column("id", Text, primary_key=True),
        Column("name", Text, nullable=False),
        Column("description", Text),
        Column("definition", JSONB, nullable=False),
        Column("inputs_example", JSONB),
        Column("valid", Boolean, nullable=False),
        Column("issues", JSONB, nullable=False),
        Column("builtin", Boolean, nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        schema=SCHEMA,
    )
    op.create_table(
        "runs",
        Column("id", UUID(as_uuid=True), primary_key=True),
        Column("goal", Text),
        Column("workflow_id", Text, ForeignKey(f"{SCHEMA}.workflows.id")),
        Column("plan", JSONB, nullable=False),
        Column("inputs", JSONB, nullable=False),
        Column("status", Text, nullable=False),
        Column("outputs", JSONB),
        Column("principal", Text, nullable=False),
        Column("role", Text, nullable=False),
        Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("finished_at", DateTime(timezone=True)),
        schema=SCHEMA,
    )
    op.create_index("ix_runs_started_at", "runs", ["started_at"], schema=SCHEMA)
    op.create_table(
        "run_nodes",
        Column("run_id", UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.runs.id", ondelete="CASCADE"), primary_key=True),
        Column("node_id", Text, primary_key=True),
        Column("item_key", Text, primary_key=True, nullable=False),
        Column("attempt", Integer, primary_key=True, nullable=False),
        Column("capability", Text, nullable=False),
        Column("implementation_key", Text),
        Column("status", Text, nullable=False),
        Column("input", JSONB),
        Column("output", JSONB),
        Column("error", Text),
        Column("started_at", DateTime(timezone=True)),
        Column("finished_at", DateTime(timezone=True)),
        Column("duration_ms", Integer),
        schema=SCHEMA,
    )
    op.create_table(
        "approvals",
        Column("id", UUID(as_uuid=True), primary_key=True),
        Column("run_id", UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.runs.id", ondelete="CASCADE"), nullable=False),
        Column("node_id", Text, nullable=False),
        Column("capability", Text, nullable=False),
        Column("payload_hash", Text, nullable=False),
        Column("requested_by", Text, nullable=False),
        Column("requested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("decided_by", Text),
        Column("decision", Text),
        Column("decided_at", DateTime(timezone=True)),
        Column("comment", Text),
        schema=SCHEMA,
    )
    op.create_table(
        "audit_log",
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("invocation_id", UUID(as_uuid=True), nullable=False),
        Column("run_id", UUID(as_uuid=True)),
        Column("principal", Text, nullable=False),
        Column("role", Text, nullable=False),
        Column("capability", Text, nullable=False),
        Column("implementation_key", Text),
        Column("status", Text, nullable=False),
        Column("reason", Text),
        Column("duration_ms", Integer),
        Column("checks", JSONB, nullable=False),
        schema=SCHEMA,
    )
    op.create_index("ix_audit_at", "audit_log", ["at"], schema=SCHEMA)
    op.create_index("ix_audit_capability", "audit_log", ["capability"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("audit_log", schema=SCHEMA)
    op.drop_table("approvals", schema=SCHEMA)
    op.drop_table("run_nodes", schema=SCHEMA)
    op.drop_table("runs", schema=SCHEMA)
    op.drop_table("workflows", schema=SCHEMA)
    op.drop_table("implementations", schema=SCHEMA)
    op.drop_table("capabilities", schema=SCHEMA)
