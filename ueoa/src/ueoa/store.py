"""Small document store for platform-managed objects (hosted agent definitions, workflow templates, ...).

Memory-only by default; with a directory it persists one JSON file per document under <dir>/<kind>/<id>.json.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")


class StoreError(ValueError):
    pass


def _check(name: str, what: str) -> str:
    if not _NAME.fullmatch(name) or ".." in name:
        raise StoreError(f"invalid {what} '{name}'")
    return name


class JsonStore:
    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory
        self._docs: dict[str, dict[str, dict[str, Any]]] = {}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            for kind_dir in directory.iterdir():
                if not kind_dir.is_dir() or not _NAME.fullmatch(kind_dir.name):
                    continue
                for path in kind_dir.glob("*.json"):
                    if _NAME.fullmatch(path.stem):
                        self._docs.setdefault(kind_dir.name, {})[path.stem] = json.loads(
                            path.read_text(encoding="utf-8")
                        )

    @property
    def persistent(self) -> bool:
        return self._dir is not None

    def list(self, kind: str) -> list[dict[str, Any]]:
        docs = self._docs.get(_check(kind, "kind"), {})
        return [copy.deepcopy(docs[key]) for key in sorted(docs)]

    def get(self, kind: str, doc_id: str) -> dict[str, Any] | None:
        doc = self._docs.get(_check(kind, "kind"), {}).get(_check(doc_id, "id"))
        return copy.deepcopy(doc) if doc is not None else None

    def put(self, kind: str, doc_id: str, doc: dict[str, Any]) -> None:
        _check(kind, "kind")
        _check(doc_id, "id")
        self._docs.setdefault(kind, {})[doc_id] = copy.deepcopy(doc)
        if self._dir is None:
            return
        folder = self._dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, ensure_ascii=False)
        Path(tmp).replace(folder / f"{doc_id}.json")

    def delete(self, kind: str, doc_id: str) -> bool:
        removed = self._docs.get(_check(kind, "kind"), {}).pop(_check(doc_id, "id"), None) is not None
        if removed and self._dir is not None:
            (self._dir / kind / f"{doc_id}.json").unlink(missing_ok=True)
        return removed
