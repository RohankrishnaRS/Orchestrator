"""Best-effort asynchronous persistence for state changes on hot paths."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)
_tasks: set[asyncio.Task[None]] = set()


def schedule_write(write: Coroutine[Any, Any, None], operation: str) -> None:
    task = asyncio.create_task(write, name=f"persist-{operation}")
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    task.add_done_callback(lambda completed: _log_failure(completed, operation))


def _log_failure(task: asyncio.Task[None], operation: str) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "background persistence failed for %s (%s)",
            operation,
            type(error).__name__,
        )


async def flush_writes() -> None:
    if _tasks:
        await asyncio.gather(*tuple(_tasks), return_exceptions=True)
