"""Persistence backends: in-memory (default) or PostgreSQL when UEOA_STORE_URL is set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ueoa.persistence.settings import StoreSettings
from ueoa.persistence.writebehind import flush_writes

if TYPE_CHECKING:
    from ueoa.persistence.postgres.engine import Database
    from ueoa.persistence.postgres.repositories import (
        ApprovalRepository,
        AuditRepository,
        CapabilityRepository,
        ImplementationRepository,
        RunRepository,
        WorkflowRepository,
    )


@dataclass
class Stores:
    backend: str
    db: Database | None = None
    audit: AuditRepository | None = None
    implementations: ImplementationRepository | None = None
    capabilities: CapabilityRepository | None = None
    workflows: WorkflowRepository | None = None
    runs: RunRepository | None = None
    approvals: ApprovalRepository | None = None

    async def healthy(self) -> bool:
        return True if self.db is None else await self.db.ping()

    async def close(self) -> None:
        await flush_writes()
        if self.db is not None:
            await self.db.dispose()


def build_stores(settings: StoreSettings | None = None) -> Stores:
    settings = settings or StoreSettings.from_env()
    if not settings.url:
        return Stores(backend="memory")
    from ueoa.persistence.postgres.engine import Database
    from ueoa.persistence.postgres.repositories import (
        ApprovalRepository,
        AuditRepository,
        CapabilityRepository,
        ImplementationRepository,
        RunRepository,
        WorkflowRepository,
    )

    db = Database(settings)
    return Stores(
        backend="postgres",
        db=db,
        audit=AuditRepository(db),
        implementations=ImplementationRepository(db),
        capabilities=CapabilityRepository(db),
        workflows=WorkflowRepository(db),
        runs=RunRepository(db),
        approvals=ApprovalRepository(db),
    )