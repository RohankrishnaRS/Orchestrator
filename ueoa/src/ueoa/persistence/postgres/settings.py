"""Persistence settings: UEOA_STORE_URL selects the backend (empty = in-memory)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StoreSettings:
    url: str
    echo_sql: bool = False
    pool_size: int = 5

    @classmethod
    def from_env(cls) -> StoreSettings:
        return cls(
            url=os.environ.get("UEOA_STORE_URL", "").strip(),
            echo_sql=os.environ.get("UEOA_STORE_ECHO_SQL", "").lower() == "true",
            pool_size=int(os.environ.get("UEOA_STORE_POOL_SIZE", "5")),
        )

    @property
    def backend(self) -> str:
        return "postgres" if self.url else "memory"