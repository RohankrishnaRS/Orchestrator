from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import CAPABILITY_ID_PATTERN, SEMVER_PATTERN, Effect

NODE_ID_PATTERN = r"^[a-z][a-z0-9_]*$"
MAX_WORKFLOW_NODES = 25  # same as orchestrator.validation.DEFAULT_MAX_NODES
MAX_NODE_TIMEOUT_S = 900


class NodeType(StrEnum):
    CAPABILITY = "capability"
    HUMAN_INPUT = "human_input"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_s: list[float] = Field(default_factory=list)
    retry_on: list[Literal["timeout", "unavailable", "rate_limited", "bad_output", "remote_error"]] = Field(
        default_factory=lambda: ["timeout", "unavailable"]
    )


class ApprovalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = True
    binds_to: Literal["payload_hash"] = "payload_hash"


class InputSpec(BaseModel):
    """Declared workflow input; JSON-Schema-like keywords (minItems, ...) are kept as extras."""

    model_config = ConfigDict(extra="allow")

    type: str
    default: Any = None
    description: str | None = None


class CanvasPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=NODE_ID_PATTERN)
    type: NodeType = NodeType.CAPABILITY
    kind: Literal["agent", "skill", "mcp_tool"] | None = None
    skill_id: str | None = Field(default=None, min_length=1, max_length=120)
    canvas_position: CanvasPosition | None = None
    capability: str | None = Field(default=None, pattern=CAPABILITY_ID_PATTERN)
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    for_each: str | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=16)
    on_item_error: Literal["fail", "continue"] = "fail"  # for_each only: "continue" keeps succeeded items
    condition: str | None = None
    effect: Effect = Effect.READ
    approval: ApprovalPolicy | None = None
    idempotency_key: str | None = None
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_s: int | None = Field(default=None, gt=0, le=MAX_NODE_TIMEOUT_S)
    prompt_key: str | None = None
    input_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_node_kind(self) -> "WorkflowNode":
        if self.type is NodeType.CAPABILITY and not self.capability:
            raise ValueError(f"node '{self.id}': capability nodes need 'capability'")
        if self.type is NodeType.HUMAN_INPUT:
            if self.capability:
                raise ValueError(f"node '{self.id}': human_input nodes cannot reference a capability")
            if self.input_schema is None:
                raise ValueError(f"node '{self.id}': human_input nodes need 'input_schema'")
            if self.effect is Effect.WRITE:
                raise ValueError(f"node '{self.id}': human_input nodes cannot write")
        return self


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    version: str = Field(pattern=SEMVER_PATTERN)
    goal: str
    allowed_tasks: list[str] | None = Field(default=None, min_length=1, max_length=50)
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=MAX_WORKFLOW_NODES)
    outputs: dict[str, str] = Field(default_factory=dict)

    def node(self, node_id: str) -> WorkflowNode:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(node_id)
