"""Run database migrations: python -m ueoa.persistence.migrate [upgrade|downgrade|current] [revision]."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config


def _load_dotenv() -> None:
    """Pick up UEOA_STORE_URL from the nearest .env when not already set."""
    if os.environ.get("UEOA_STORE_URL"):
        return
    for folder in (Path.cwd(), *Path.cwd().parents):
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("UEOA_STORE_URL="):
                    os.environ["UEOA_STORE_URL"] = line.split("=", 1)[1].strip().strip('"')
            return


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = list(sys.argv[1:] if argv is None else argv)
    action = args[0] if args else "upgrade"
    revision = args[1] if len(args) > 1 else ("head" if action == "upgrade" else "-1")
    cfg = Config(str(Path(__file__).parent / "postgres" / "alembic.ini"))
    if action == "upgrade":
        command.upgrade(cfg, revision)
    elif action == "downgrade":
        command.downgrade(cfg, revision)
    elif action == "current":
        command.current(cfg, verbose=True)
    else:
        print("usage: python -m ueoa.persistence.migrate [upgrade|downgrade|current] [revision]")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
