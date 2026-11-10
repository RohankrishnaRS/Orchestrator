"""PostgreSQL repositories for platform state."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from ueoa.contracts import CapabilityContract
from ueoa.orchestrator.runner import PendingApproval, Run
from ueoa.persistence.postgres.engine import Database
from ueoa.persistence.postgres.models import (
    ApprovalRow,
    AuditRow,
    CapabilityRow,
    ImplementationRow,
    RunNodeRow,
    RunRow,
    WorkflowRow,
)
from ueoa.registry.models import AgentCard, CheckResult, Implementation, ImplementationStatus

_OPERATOR_STATUSES = [
    ImplementationStatus.APPROVED.value,
    ImplementationStatus.DEPRECATED.value,
    ImplementationStatus.REVOKED.value,
]


class AuditRepository:
    def __init__(self, db: Database):
        self._db = db

    async def append(self, entry: dict[str, Any]) -> None:
        row = AuditRow(
            invocation_id=uuid.UUID(str(entry["invocation_id"])),
            run_id=uuid.UUID(str(entry["run_id"])) if entry.get("run_id") else None,
            at=datetime.fromisoformat(
                str(entry.get("at") or datetime.now().astimezone().isoformat())
            ),
            principal=str(entry["principal"]),
            role=str(entry["role"]),
            capability=str(entry["capability"]),
            implementation_key=entry.get("implementation"),
            status=str(entry["status"]),
            reason=entry.get("reason"),
            duration_ms=entry.get("duration_ms"),
            checks=list(entry.get("checks") or []),
            output=entry.get("output"),
        )
        async with self._db.session() as session:
            session.add(row)

    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.session() as session:
            rows = (await session.execute(select(AuditRow).order_by(AuditRow.at.desc()).limit(limit))).scalars()
            return [
                {
                    "at": r.at.isoformat(),
                    "invocation_id": str(r.invocation_id),
                    "run_id": str(r.run_id) if r.run_id else None,
                    "principal": r.principal,
                    "role": r.role,
                    "capability": r.capability,
                    "implementation": r.implementation_key,
                    "status": r.status,
                    "reason": r.reason,
                    "duration_ms": r.duration_ms,
                    "checks": r.checks,
                    "output": r.output,
                }
                for r in rows
            ]


class ImplementationRepository:
    def __init__(self, db: Database):
        self._db = db
        self._locks: dict[str, asyncio.Lock] = {}

    async def upsert(self, impl: Implementation, *, seed: bool = False) -> None:
        lock = self._locks.setdefault(impl.key, asyncio.Lock())
        async with lock:
            await self._upsert(impl, seed=seed)

    async def _upsert(self, impl: Implementation, *, seed: bool) -> None:
        ext = impl.card.ueoa
        values = {
            "key": impl.key,
            "agent_id": ext.id,
            "version": impl.card.version,
            "binding": ext.binding,
            "kind": ext.kind,
            "status": impl.status.value,
            "priority": impl.priority,
            "source": impl.source,
            "card": impl.card.model_dump(mode="json", by_alias=True, exclude_none=True),
            "card_sha256": impl.card_sha256,
            "implements": ext.implements,
            "checks": [check.model_dump(mode="json") for check in impl.checks],
        }
        statement = insert(ImplementationRow).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[ImplementationRow.key],
            set_={key: getattr(statement.excluded, key) for key in values if key != "key"},
            where=(
                ImplementationRow.status.not_in(_OPERATOR_STATUSES)
                if seed
                else None
            ),
        )
        async with self._db.session() as session:
            await session.execute(statement)

    async def set_status(self, key: str, status: ImplementationStatus, priority: int | None = None) -> None:
        values: dict[str, Any] = {"status": status.value}
        if priority is not None:
            values["priority"] = priority
        async with self._db.session() as session:
            await session.execute(update(ImplementationRow).where(ImplementationRow.key == key).values(**values))

    async def get(self, key: str) -> Implementation | None:
        async with self._db.session() as session:
            row = await session.get(ImplementationRow, key)
            return _implementation(row) if row is not None else None

    async def list(self) -> list[Implementation]:
        async with self._db.session() as session:
            rows = (await session.execute(select(ImplementationRow).order_by(ImplementationRow.key))).scalars()
            return [_implementation(row) for row in rows]


def _implementation(row: ImplementationRow) -> Implementation:
    card = AgentCard.model_validate(row.card)
    return Implementation(
        card=card,
        status=ImplementationStatus(row.status),
        priority=row.priority,
        source=row.source,
        card_sha256=row.card_sha256,
        checks=[CheckResult.model_validate(check) for check in row.checks],
        registered_at=row.registered_at,
    )


class CapabilityRepository:
    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, capability: CapabilityContract) -> None:
        contract = capability.model_dump(mode="json")
        statement = insert(CapabilityRow).values(
            id=capability.id,
            version=capability.version,
            major=int(capability.version.split(".", 1)[0]),
            effects=str(capability.effects),
            contract=contract,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[CapabilityRow.id, CapabilityRow.version],
            set_={"major": statement.excluded.major, "effects": statement.excluded.effects, "contract": contract},
        )
        async with self._db.session() as session:
            await session.execute(statement)

    async def list(self) -> list[CapabilityContract]:
        async with self._db.session() as session:
            rows = (await session.execute(select(CapabilityRow).order_by(CapabilityRow.id))).scalars()
            return [CapabilityContract.model_validate(row.contract) for row in rows]


class WorkflowRepository:
    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, doc: dict[str, Any], *, kind: str = "workflows") -> None:
        metadata = {key: value for key, value in doc.items() if key not in {
            "id", "name", "description", "workflow", "inputs_example", "valid", "issues", "builtin"
        }}
        metadata["_kind"] = kind
        statement = insert(WorkflowRow).values(
            id=str(doc["id"]),
            name=str(doc.get("name", doc["id"])),
            description=doc.get("description"),
            definition=dict(doc["workflow"]),
            inputs_example=doc.get("inputs_example"),
            valid=bool(doc.get("valid", True)),
            issues=list(doc.get("issues") or []),
            builtin=bool(doc.get("builtin", False)),
            extra_data=metadata,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[WorkflowRow.id],
            set_={
                "name": statement.excluded.name,
                "description": statement.excluded.description,
                "definition": statement.excluded.definition,
                "inputs_example": statement.excluded.inputs_example,
                "valid": statement.excluded.valid,
                "issues": statement.excluded.issues,
                "builtin": statement.excluded.builtin,
                "extra_data": statement.excluded.extra_data,
            },
        )
        async with self._db.session() as session:
            await session.execute(statement)

    async def get(self, workflow_id: str) -> dict[str, Any] | None:
        async with self._db.session() as session:
            row = await session.get(WorkflowRow, workflow_id)
            return _workflow(row) if row is not None else None

    async def list(self) -> list[dict[str, Any]]:
        async with self._db.session() as session:
            rows = (await session.execute(select(WorkflowRow).order_by(WorkflowRow.id))).scalars()
            return [_workflow(row) for row in rows]

    async def delete(self, workflow_id: str) -> None:
        async with self._db.session() as session:
            row = await session.get(WorkflowRow, workflow_id)
            if row is not None:
                await session.delete(row)


def _workflow(row: WorkflowRow) -> dict[str, Any]:
    return {
        **row.extra_data,
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "workflow": row.definition,
        "inputs_example": row.inputs_example or {},
        "valid": row.valid,
        "issues": row.issues,
        "builtin": row.builtin,
    }


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class RunRepository:
    def __init__(self, db: Database):
        self._db = db
        self._locks: dict[str, asyncio.Lock] = {}

    async def create(self, run: Run) -> None:
        row = RunRow(
            id=uuid.UUID(run.id),
            goal=run.spec.goal,
            workflow_id=None,
            plan=run.plan or {},
            workflow=run.spec.model_dump(mode="json", exclude_none=True),
            inputs=run.inputs,
            status=str(run.status),
            outputs=run.outputs,
            error=run.error,
            principal=run.principal.name,
            role=run.principal.role,
            started_at=_datetime(run.created_at),
            finished_at=_datetime(run.finished_at),
        )
        async with self._db.session() as session:
            session.add(row)

    async def persist(self, run: Run) -> None:
        lock = self._locks.setdefault(run.id, asyncio.Lock())
        async with lock:
            await self._persist(run)

    async def _persist(self, run: Run) -> None:
        run_values = {
            "id": uuid.UUID(run.id),
            "goal": run.spec.goal,
            "workflow_id": None,
            "plan": run.plan or {},
            "workflow": run.spec.model_dump(mode="json", exclude_none=True),
            "inputs": run.inputs,
            "status": str(run.status),
            "outputs": run.outputs,
            "error": run.error,
            "principal": run.principal.name,
            "role": run.principal.role,
            "started_at": _datetime(run.created_at),
            "finished_at": _datetime(run.finished_at),
        }
        run_insert = insert(RunRow).values(**run_values).on_conflict_do_update(
            index_elements=[RunRow.id],
            set_={key: value for key, value in run_values.items() if key != "id"},
        )
        async with self._db.session() as session:
            await session.execute(run_insert)
            for node_id, result in run.nodes.items():
                node = run.spec.node(node_id)
                snapshot = asdict(result)
                await _upsert_run_node(
                    session,
                    run_id=run.id,
                    node_id=node_id,
                    capability=node.capability or "",
                    result=snapshot,
                    attempt=max(1, result.attempts),
                )
                for item in result.items:
                    await _upsert_run_node(
                        session,
                        run_id=run.id,
                        node_id=node_id,
                        capability=node.capability or "",
                        result=asdict(item),
                        item_key=str(item.index),
                        attempt=max(1, item.attempts),
                    )
            for pending in run.approvals.values():
                item_key = "" if pending.item is None else str(pending.item)
                approval_id = uuid.uuid5(uuid.UUID(run.id), f"{pending.node}:{item_key}:{pending.payload_hash}")
                approval_insert = insert(ApprovalRow).values(
                    id=approval_id,
                    run_id=uuid.UUID(run.id),
                    node_id=pending.node,
                    item_key=item_key,
                    capability=pending.capability,
                    payload_hash=pending.payload_hash,
                    payload=pending.payload,
                    idempotency_key=pending.idempotency_key,
                    requested_by=pending.requested_by,
                    requested_at=_datetime(pending.requested_at),
                ).on_conflict_do_nothing(index_elements=[ApprovalRow.id])
                await session.execute(approval_insert)

    async def update_status(self, run_id: str, status: str) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(RunRow).where(RunRow.id == uuid.UUID(run_id)).values(status=status)
            )

    async def update_outputs(self, run_id: str, outputs: dict[str, Any] | None) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(RunRow).where(RunRow.id == uuid.UUID(run_id)).values(outputs=outputs)
            )

    async def update_finished_at(self, run_id: str, finished_at: str | None) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(RunRow).where(RunRow.id == uuid.UUID(run_id)).values(finished_at=_datetime(finished_at))
            )

    async def upsert_node(
        self,
        run_id: str,
        node_id: str,
        capability: str,
        result: dict[str, Any],
        *,
        item_key: str = "",
        attempt: int = 1,
    ) -> None:
        async with self._db.session() as session:
            await _upsert_run_node(
                session,
                run_id=run_id,
                node_id=node_id,
                capability=capability,
                result=result,
                item_key=item_key,
                attempt=attempt,
            )

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._db.session() as session:
            row = await session.get(RunRow, uuid.UUID(run_id))
            if row is None:
                return None
            nodes = (
                await session.execute(
                    select(RunNodeRow)
                    .where(RunNodeRow.run_id == row.id, RunNodeRow.item_key == "")
                    .order_by(RunNodeRow.node_id, RunNodeRow.attempt.desc())
                )
            ).scalars()
            approvals = (
                await session.execute(
                    select(ApprovalRow).where(ApprovalRow.run_id == row.id, ApprovalRow.decision.is_(None))
                )
            ).scalars()
            return _run_record(row, nodes, approvals)

    async def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.session() as session:
            rows = list(
                (
                    await session.execute(select(RunRow).order_by(RunRow.started_at.desc()).limit(limit))
                ).scalars()
            )
            if not rows:
                return []
            run_ids = [row.id for row in rows]
            nodes = list(
                (
                    await session.execute(
                        select(RunNodeRow)
                        .where(RunNodeRow.run_id.in_(run_ids), RunNodeRow.item_key == "")
                        .order_by(RunNodeRow.node_id, RunNodeRow.attempt.desc())
                    )
                ).scalars()
            )
            approvals = list(
                (
                    await session.execute(
                        select(ApprovalRow).where(
                            ApprovalRow.run_id.in_(run_ids), ApprovalRow.decision.is_(None)
                        )
                    )
                ).scalars()
            )
            nodes_by_run: dict[uuid.UUID, list[RunNodeRow]] = {}
            approvals_by_run: dict[uuid.UUID, list[ApprovalRow]] = {}
            for node in nodes:
                nodes_by_run.setdefault(node.run_id, []).append(node)
            for approval in approvals:
                approvals_by_run.setdefault(approval.run_id, []).append(approval)
            return [
                _run_record(row, nodes_by_run.get(row.id, []), approvals_by_run.get(row.id, []))
                for row in rows
            ]


async def _upsert_run_node(
    session: Any,
    *,
    run_id: str,
    node_id: str,
    capability: str,
    result: dict[str, Any],
    item_key: str = "",
    attempt: int = 1,
) -> None:
    statement = insert(RunNodeRow).values(
        run_id=uuid.UUID(run_id),
        node_id=node_id,
        item_key=item_key,
        attempt=max(1, attempt),
        capability=capability,
        implementation_key=result.get("implementation"),
        status=str(result.get("state", "pending")),
        input=result.get("input") if isinstance(result.get("input"), dict) else None,
        output=result.get("output"),
        error=result.get("error"),
        started_at=_datetime(result.get("started_at")),
        finished_at=_datetime(result.get("finished_at")),
        duration_ms=None,
        item_data=result,
    )
    statement = statement.on_conflict_do_update(
        index_elements=[RunNodeRow.run_id, RunNodeRow.node_id, RunNodeRow.item_key, RunNodeRow.attempt],
        set_={
            "capability": statement.excluded.capability,
            "implementation_key": statement.excluded.implementation_key,
            "status": statement.excluded.status,
            "input": statement.excluded.input,
            "output": statement.excluded.output,
            "error": statement.excluded.error,
            "started_at": statement.excluded.started_at,
            "finished_at": statement.excluded.finished_at,
            "item_data": statement.excluded.item_data,
        },
    )
    await session.execute(statement)


def _run_record(row: RunRow, nodes: Any, approvals: Any) -> dict[str, Any]:
    latest_nodes: dict[str, dict[str, Any]] = {}
    for node in nodes:
        latest_nodes.setdefault(
            node.node_id,
            node.item_data
            or {
                "state": node.status,
                "output": node.output,
                "error": node.error,
                "started_at": node.started_at.isoformat() if node.started_at else None,
                "finished_at": node.finished_at.isoformat() if node.finished_at else None,
            },
        )
    return {
        "id": str(row.id),
        "goal": row.goal,
        "workflow": row.workflow,
        "plan": row.plan,
        "inputs": row.inputs,
        "status": row.status,
        "outputs": row.outputs,
        "error": row.error,
        "principal": row.principal,
        "role": row.role,
        "created_at": row.started_at.isoformat(),
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "nodes": latest_nodes,
        "approvals": [
            {
                "node": approval.node_id,
                "item": int(approval.item_key) if approval.item_key else None,
                "capability": approval.capability,
                "payload_hash": approval.payload_hash,
                "payload": approval.payload,
                "idempotency_key": approval.idempotency_key,
                "requested_at": approval.requested_at.isoformat(),
                "requested_by": approval.requested_by,
            }
            for approval in approvals
        ],
    }


class ApprovalRepository:
    def __init__(self, db: Database):
        self._db = db

    async def create(self, run_id: str, pending: PendingApproval) -> None:
        item_key = "" if pending.item is None else str(pending.item)
        session_id = uuid.uuid5(uuid.UUID(run_id), f"{pending.node}:{item_key}:{pending.payload_hash}")
        statement = insert(ApprovalRow).values(
            id=session_id,
            run_id=uuid.UUID(run_id),
            node_id=pending.node,
            item_key=item_key,
            capability=pending.capability,
            payload_hash=pending.payload_hash,
            payload=pending.payload,
            idempotency_key=pending.idempotency_key,
            requested_by=pending.requested_by,
            requested_at=_datetime(pending.requested_at),
        ).on_conflict_do_nothing(index_elements=[ApprovalRow.id])
        async with self._db.session() as session:
            await session.execute(statement)

    async def decide(
        self, run_id: str, node_id: str, item: int | None, decision: str, decided_by: str
    ) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(ApprovalRow)
                .where(
                    ApprovalRow.run_id == uuid.UUID(run_id),
                    ApprovalRow.node_id == node_id,
                    ApprovalRow.item_key == ("" if item is None else str(item)),
                    ApprovalRow.decision.is_(None),
                )
                .values(decision=decision, decided_by=decided_by)
            )

    async def list_pending(self) -> list[dict[str, Any]]:
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(ApprovalRow).where(ApprovalRow.decision.is_(None)).order_by(ApprovalRow.requested_at)
                )
            ).scalars()
            return [
                {
                    "run_id": str(row.run_id),
                    "node": row.node_id,
                    "item": int(row.item_key) if row.item_key else None,
                    "capability": row.capability,
                    "payload": row.payload,
                    "payload_hash": row.payload_hash,
                    "idempotency_key": row.idempotency_key,
                    "requested_at": row.requested_at.isoformat(),
                    "requested_by": row.requested_by,
                }
                for row in rows
            ]
