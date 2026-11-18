import logging

import pytest

from ueoa.persistence.writebehind import flush_writes, schedule_write


@pytest.mark.anyio
async def test_write_behind_logs_database_errors_without_propagating(caplog: pytest.LogCaptureFixture) -> None:
    async def fail() -> None:
        raise RuntimeError("database unavailable")

    with caplog.at_level(logging.ERROR, logger="ueoa.persistence.writebehind"):
        schedule_write(fail(), "unit-test")
        await flush_writes()

    assert "background persistence failed for unit-test" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "database unavailable" not in caplog.text
