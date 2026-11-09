"""Persistence backends: in-memory (default) or PostgreSQL when UEOA_STORE_URL is set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ueoa.persistence.settings import StoreSettings

if TYPE_CHECKING:
    from ueoa.persistence.postgres.engine import Database
    from ueoa.persistence.postgres.repositories import AuditRepository


@dataclass
class Stores:
    backend: str
    db: Database | None = None
    audit: AuditRepository | None = None

    async def healthy(self) -> bool:
        return True if self.db is None else await self.db.ping()

    async def close(self) -> None:
        if self.db is not None:
            await self.db.dispose()


def build_stores(settings: StoreSettings | None = None) -> Stores:
    settings = settings or StoreSettings.from_env()
    if not settings.url:
        return Stores(backend="memory")
    from ueoa.persistence.postgres.engine import Database
    from ueoa.persistence.postgres.repositories import AuditRepository

    db = Database(settings)
    return Stores(backend="postgres", db=db, audit=AuditRepository(db))