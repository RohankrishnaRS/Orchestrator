"""Static analysis of a WorkflowSpec: expressions per node, dependency graph, ancestors, order."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ueoa.contracts import WorkflowNode, WorkflowSpec

from .expressions import RefUse, is_expression, parse, references


@dataclass(frozen=True)
class NodeExpression:
    location: str  # e.g. "condition", "for_each", "inputs.documents"
    source: str


def _binding_expressions(value: Any, location: str) -> Iterator[NodeExpression]:
    if is_expression(value):
        yield NodeExpression(location, str(value))
    elif isinstance(value, dict):
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            yield from _binding_expressions(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            yield from _binding_expressions(item, f"{location}[{index}]")


def node_expressions(node: WorkflowNode) -> list[NodeExpression]:
    found: list[NodeExpression] = []
    if node.condition is not None:
        found.append(NodeExpression("condition", node.condition))
    if node.for_each is not None:
        found.append(NodeExpression("for_each", node.for_each))
    found.extend(_binding_expressions(node.inputs, "inputs"))
    return found


def node_ref_uses(node: WorkflowNode) -> list[RefUse]:
    """All references used by a node. Raises ExpressionError on invalid syntax."""
    uses: list[RefUse] = []
    for expression in node_expressions(node):
        uses.extend(references(parse(expression.source)))
    return uses


def referenced_nodes(node: WorkflowNode) -> dict[str, bool]:
    """Referenced node id -> True if *every* reference to it is guarded (default/exists)."""
    result: dict[str, bool] = {}
    for use in node_ref_uses(node):
        if use.ref.root != "nodes" or not use.ref.path:
            continue
        target = str(use.ref.path[0])
        result[target] = result.get(target, True) and use.guarded
    return result


class CycleError(ValueError):
    def __init__(self, nodes: list[str]) -> None:
        super().__init__(f"dependency cycle among: {', '.join(sorted(nodes))}")
        self.nodes = nodes


def topological_order(spec: WorkflowSpec) -> list[str]:
    """Kahn's algorithm; stable by declaration order. Raises CycleError."""
    ids = [node.id for node in spec.nodes]
    known = set(ids)
    indegree = {node_id: 0 for node_id in ids}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for node in spec.nodes:
        for dep in node.depends_on:
            if dep in known:
                indegree[node.id] += 1
                dependents[dep].append(node.id)
    queue = [node_id for node_id in ids if indegree[node_id] == 0]
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in dependents[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(ids):
        raise CycleError([node_id for node_id in ids if node_id not in order])
    return order


def ancestors(spec: WorkflowSpec) -> dict[str, set[str]]:
    """node id -> all transitive dependencies. Assumes the graph is acyclic."""
    result: dict[str, set[str]] = {}
    for node_id in topological_order(spec):
        node = spec.node(node_id)
        found: set[str] = set()
        for dep in node.depends_on:
            found.add(dep)
            found |= result.get(dep, set())
        result[node_id] = found
    return result


def descendants(spec: WorkflowSpec, node_ids: set[str]) -> set[str]:
    """All nodes that transitively depend on any of node_ids (excluding node_ids)."""
    anc = ancestors(spec)
    return {node_id for node_id, deps in anc.items() if deps & node_ids}


def topological_levels(spec: WorkflowSpec) -> list[list[str]]:
    """Groups of nodes that can run in parallel, in execution order."""
    level: dict[str, int] = {}
    for node_id in topological_order(spec):
        deps = spec.node(node_id).depends_on
        level[node_id] = 1 + max((level[dep] for dep in deps), default=-1)
    grouped: dict[int, list[str]] = {}
    for node_id, index in level.items():
        grouped.setdefault(index, []).append(node_id)
    return [grouped[index] for index in sorted(grouped)]
