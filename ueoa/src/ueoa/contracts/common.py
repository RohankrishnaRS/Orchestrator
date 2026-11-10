from enum import StrEnum

SEMVER_PATTERN = r"^\d+\.\d+\.\d+$"
CAPABILITY_ID_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
