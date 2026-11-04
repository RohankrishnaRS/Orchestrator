"""Pure scheduling logic: node states, the ready set, skip propagation and run status.

The run controller keeps node states and calls these functions; nothing here does I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from ueoa.contracts import WorkflowSpec

from .graph import descendants, referenced_nodes


class NodeState(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_INPUT = "waiting_input"
    WAITING_APPROVAL = "waiting_approval"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    COMPLETED_REJECTED = "completed_rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


SATISFIED = frozenset({NodeState.SUCCEEDED, NodeState.SKIPPED})
TERMINAL = frozenset(
    {NodeState.SUCCEEDED, NodeState.FAILED, NodeState.SKIPPED, NodeState.REJECTED, NodeState.CANCELLED}
)
BLOCKING = frozenset({NodeState.FAILED, NodeState.REJECTED, NodeState.CANCELLED})
IN_PROGRESS = frozenset({NodeState.READY, NodeState.RUNNING})
WAITING = frozenset({NodeState.WAITING_INPUT, NodeState.WAITING_APPROVAL})


def initial_states(spec: WorkflowSpec) -> dict[str, NodeState]:
    return {node.id: NodeState.PENDING for node in spec.nodes}


def must_skip(spec: WorkflowSpec, node_id: str, states: Mapping[str, NodeState]) -> bool:
    """A node is skipped when it references a skipped node (any upstream node) without a default/exists guard."""
    refs = referenced_nodes(spec.node(node_id))
    return any(not guarded and states.get(ref) is NodeState.SKIPPED for ref, guarded in refs.items())


def eligible_nodes(spec: WorkflowSpec, states: Mapping[str, NodeState]) -> list[str]:
    """Pending nodes whose dependencies are all satisfied. Conditions are evaluated by the controller."""
    return [
        node.id
        for node in spec.nodes
        if states.get(node.id) is NodeState.PENDING and all(states.get(dep) in SATISFIED for dep in node.depends_on)
    ]


def blocked_nodes(spec: WorkflowSpec, states: Mapping[str, NodeState]) -> list[str]:
    """Pending nodes that cannot run until a failed / rejected / cancelled ancestor changes."""
    stuck = {node_id for node_id, state in states.items() if state in BLOCKING}
    if not stuck:
        return []
    downstream = descendants(spec, stuck)
    return [node.id for node in spec.nodes if node.id in downstream and states.get(node.id) is NodeState.PENDING]


def aggregate_item_states(item_states: Iterable[NodeState]) -> NodeState:
    """Node state for a for_each node, derived from its items."""
    items = list(item_states)
    if not items or all(state is NodeState.SUCCEEDED for state in items):
        return NodeState.SUCCEEDED
    if any(state in IN_PROGRESS or state is NodeState.PENDING for state in items):
        return NodeState.RUNNING
    if any(state is NodeState.WAITING_APPROVAL for state in items):
        return NodeState.WAITING_APPROVAL
    if any(state is NodeState.FAILED for state in items):
        return NodeState.FAILED
    if all(state in {NodeState.SUCCEEDED, NodeState.REJECTED} for state in items):
        return NodeState.REJECTED if all(s is NodeState.REJECTED for s in items) else NodeState.SUCCEEDED
    return NodeState.CANCELLED


def run_status(spec: WorkflowSpec, states: Mapping[str, NodeState]) -> RunStatus:
    values = [states.get(node.id, NodeState.PENDING) for node in spec.nodes]
    if any(state is NodeState.CANCELLED for state in values) and not any(s in IN_PROGRESS for s in values):
        return RunStatus.CANCELLED
    if any(state in IN_PROGRESS for state in values) or eligible_nodes(spec, states):
        return RunStatus.RUNNING
    if any(state in WAITING for state in values):
        return RunStatus.WAITING
    if any(state is NodeState.FAILED for state in values) or blocked_nodes(spec, states):
        return RunStatus.FAILED
    if any(state is NodeState.REJECTED for state in values):
        return RunStatus.COMPLETED_REJECTED
    return RunStatus.COMPLETED


def nodes_to_resume(states: Mapping[str, NodeState]) -> list[str]:
    """Failed nodes go back to pending on 'resume failed items'; succeeded nodes are never re-run."""
    return [node_id for node_id, state in states.items() if state is NodeState.FAILED]


def nodes_to_rerun(spec: WorkflowSpec, changed: set[str]) -> set[str]:
    """Refinement: the changed nodes plus everything downstream of them."""
    return changed | descendants(spec, changed)
