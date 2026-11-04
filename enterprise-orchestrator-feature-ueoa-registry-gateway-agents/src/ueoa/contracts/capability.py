from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .common import CAPABILITY_ID_PATTERN, SEMVER_PATTERN, Effect


class SelectionDoc(BaseModel):
    """What the router reads to decide whether this capability fits a task."""

    model_config = ConfigDict(extra="forbid")

    use_when: str
    do_not_use_when: str = ""
    examples: dict[str, list[str]] = Field(default_factory=dict)


class CapabilityContract(BaseModel):
    """A capability is a contract (what is needed); implementations fulfil it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=CAPABILITY_ID_PATTERN)
    version: str = Field(pattern=SEMVER_PATTERN)
    description: str
    effects: Effect
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    languages: tuple[str, ...] = ("en",)
    selection_doc: SelectionDoc | None = None
    # inputs used by conformance checks: every implementation must return a schema-valid output for each
    samples: tuple[dict[str, Any], ...] = ()
