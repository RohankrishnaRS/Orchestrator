"""PostgreSQL repositories. AuditRepository is complete; the others follow the same pattern."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from ueoa.persistence.postgres.engine import Database
from ueoa.persistence.postgres.models import AuditRow


class AuditRepository:
    def __init__(self, db: Database):
        self._db = db

    async def append(self, entry: dict[str, Any]) -> None:
        row = AuditRow(
            invocation_id=uuid.UUID(str(entry["invocation_id"])),
            run_id=uuid.UUID(str(entry["run_id"])) if entry.get("run_id") else None,
            principal=str(entry["principal"]),
            role=str(entry["role"]),
            capability=str(entry["capability"]),
            implementation_key=entry.get("implementation"),
            status=str(entry["status"]),
            reason=entry.get("reason"),
            duration_ms=entry.get("duration_ms"),
            checks=list(entry.get("checks") or []),
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
                }
                for r in rows
            ]