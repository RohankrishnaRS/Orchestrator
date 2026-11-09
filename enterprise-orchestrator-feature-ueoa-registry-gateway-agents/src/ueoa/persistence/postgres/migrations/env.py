"""Alembic environment: async engine from UEOA_STORE_URL, schema 'ueoa'."""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from ueoa.persistence.postgres.models import SCHEMA, Base

config = context.config
target_metadata = Base.metadata
url = os.environ.get("UEOA_STORE_URL", "")
if not url:
    raise SystemExit("UEOA_STORE_URL is not set")
config.set_main_option("sqlalchemy.url", url)


def include_object(obj, name, type_, reflected, compare_to):  # noqa: ANN001, D103
    return getattr(obj, "schema", SCHEMA) in (SCHEMA, None)


def do_run_migrations(connection) -> None:  # noqa: ANN001
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        version_table_schema=SCHEMA,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(config.get_section(config.config_ini_section) or {}, prefix="sqlalchemy.")
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


asyncio.run(run_migrations_online())
