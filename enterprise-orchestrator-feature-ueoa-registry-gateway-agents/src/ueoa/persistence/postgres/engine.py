"""Async SQLAlchemy engine and session factory for the PostgreSQL backend."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ueoa.persistence.settings import StoreSettings


class Database:
    def __init__(self, settings: StoreSettings):
        self.engine: AsyncEngine = create_async_engine(
            settings.url, echo=settings.echo_sql, pool_size=settings.pool_size, pool_pre_ping=True
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - health probe
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()
