from sqlalchemy.exc import OperationalError

from ueoa.persistence import migrate


def test_migration_connection_error_is_actionable_and_redacted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(migrate, "_load_dotenv", lambda: None)

    def fail_upgrade(*args, **kwargs) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(migrate.command, "upgrade", fail_upgrade)

    assert migrate.main(["upgrade"]) == 1
    output = capsys.readouterr()
    assert "Could not connect to PostgreSQL" in output.err
    assert "UEOA_STORE_URL" in output.err
    assert "connection refused" not in output.err


def test_migration_database_error_is_actionable_and_redacted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(migrate, "_load_dotenv", lambda: None)

    def fail_upgrade(*args, **kwargs) -> None:
        raise OperationalError("connect", {}, OSError("connection refused"))

    monkeypatch.setattr(migrate.command, "upgrade", fail_upgrade)

    assert migrate.main(["upgrade"]) == 1
    output = capsys.readouterr()
    assert "Could not connect to PostgreSQL" in output.err
    assert "connection refused" not in output.err
