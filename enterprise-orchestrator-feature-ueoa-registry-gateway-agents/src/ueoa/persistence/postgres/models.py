"""SQLAlchemy models for the UEOA persistence schema (PostgreSQL, schema 'ueoa')."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "ueoa"


class Base(DeclarativeBase):
    __table_args__ = {"schema": SCHEMA}


class CapabilityRow(Base):
    __tablename__ = "capabilities"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[str] = mapped_column(Text, primary_key=True)
    major: Mapped[int] = mapped_column(Integer, nullable=False)
    effects: Mapped[str] = mapped_column(Text, nullable=False)
    contract: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ImplementationRow(Base):
    __tablename__ = "implementations"
    __table_args__ = (
        Index("ix_implementations_implements", "implements", postgresql_using="gin"),
        Index("ix_implementations_status", "status"),
        {"schema": SCHEMA},
    )
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    binding: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    card: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    card_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    implements: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    checks: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class WorkflowRow(Base):
    __tablename__ = "workflows"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    definition: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    inputs_example: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issues: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_started_at", "started_at"), {"schema": SCHEMA})
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    goal: Mapped[str | None] = mapped_column(Text)
    workflow_id: Mapped[str | None] = mapped_column(Text, ForeignKey(f"{SCHEMA}.workflows.id"))
    plan: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    outputs: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    principal: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunNodeRow(Base):
    __tablename__ = "run_nodes"
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.runs.id", ondelete="CASCADE"), primary_key=True
    )
    node_id: Mapped[str] = mapped_column(Text, primary_key=True)
    item_key: Mapped[str] = mapped_column(Text, primary_key=True, default="")
    attempt: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    capability: Mapped[str] = mapped_column(Text, nullable=False)
    implementation_key: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    input: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)


class ApprovalRow(Base):
    __tablename__ = "approvals"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.runs.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str] = mapped_column(Text, nullable=False)
    capability: Mapped[str] = mapped_column(Text, nullable=False)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    requested_by: Mapped[str] = mapped_column(Text, nullable=False)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    decided_by: Mapped[str | None] = mapped_column(Text)
    decision: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    comment: Mapped[str | None] = mapped_column(Text)


class AuditRow(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_at", "at"),
        Index("ix_audit_capability", "capability"),
        {"schema": SCHEMA},
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    invocation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    principal: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    capability: Mapped[str] = mapped_column(Text, nullable=False)
    implementation_key: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    checks: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)