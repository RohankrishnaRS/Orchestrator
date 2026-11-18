"""Binding suggestions for composing workflows: what each node input can be wired to, and safe auto-wiring."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ueoa.contracts import CapabilityContract, NodeType, WorkflowNode, WorkflowSpec

from .expressions import ExpressionError, Ref, parse
from .validation import infer_reference_schema, is_compatible

# proximity ranks: lower is closer to the node being wired
_ITEM, _DIRECT, _ANCESTOR, _INPUT, _OTHER = range(5)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class _Candidate:
    expr: str
    label: str
    rank: int
    schema: dict[str, Any] | None


def _type_name(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "unknown"
    declared: Any = schema.get("type")  # pyright: ignore[reportUnknownMemberType]
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list) and declared and all(isinstance(t, str) for t in declared):  # pyright: ignore[reportUnknownVariableType]
        return "|".join(declared)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(schema.get("properties"), dict):  # pyright: ignore[reportUnknownMemberType]
        return "object"
    if isinstance(schema.get("items"), dict):  # pyright: ignore[reportUnknownMemberType]
        return "array"
    return "unknown"


def _is_array(schema: Any) -> bool:
    return "array" in _type_name(schema).split("|")


def _items(schema: Any) -> Any:
    return schema.get("items") if isinstance(schema, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _properties(schema: Any) -> dict[str, Any]:
    properties: Any = schema.get("properties") if isinstance(schema, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(properties, dict):
        return {}
    return {str(k): v for k, v in properties.items() if _NAME.fullmatch(str(k))}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _output_schema(node: WorkflowNode, capabilities: Mapping[str, CapabilityContract]) -> dict[str, Any]:
    if node.type is NodeType.HUMAN_INPUT:
        return node.input_schema or {}
    capability = capabilities.get(node.capability or "")
    return capability.output_schema if capability is not None else {}


def _reachable(spec: WorkflowSpec, start: str, *, downstream: bool) -> set[str]:
    """Transitive dependents (downstream) or dependencies of start; tolerates cycles while a spec is edited."""
    edges: dict[str, list[str]] = {}
    for node in spec.nodes:
        for dep in node.depends_on:
            if downstream:
                edges.setdefault(dep, []).append(node.id)
            else:
                edges.setdefault(node.id, []).append(dep)
    found: set[str] = set()
    pending = [start]
    while pending:
        for nxt in edges.get(pending.pop(), []):
            if nxt not in found:
                found.add(nxt)
                pending.append(nxt)
    return found


def _eligible(spec: WorkflowSpec, node: WorkflowNode) -> list[tuple[WorkflowNode, int]]:
    """Nodes this node may reference without creating a cycle, with their proximity rank, closest first."""
    excluded = _reachable(spec, node.id, downstream=True) | {node.id}
    upstream = _reachable(spec, node.id, downstream=False)
    ranked: list[tuple[WorkflowNode, int]] = []
    for other in spec.nodes:
        if other.id in excluded:
            continue
        rank = _DIRECT if other.id in node.depends_on else _ANCESTOR if other.id in upstream else _OTHER
        ranked.append((other, rank))
    return sorted(ranked, key=lambda pair: pair[1])


def _fields(base: str, who: str, schema: Any, rank: int, suffix: str = "") -> list[tuple[str, str, int]]:
    """base.<p> for each property, plus base.<p>.<q> one level into object properties."""
    found: list[tuple[str, str, int]] = []
    for name, sub in _properties(schema).items():
        found.append((f"{base}.{name}", f"{who} → {name}{suffix}", rank))
        found.extend((f"{base}.{name}.{inner}", f"{who} → {name}.{inner}{suffix}", rank) for inner in _properties(sub))
    return found


def _candidates(
    spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract], node: WorkflowNode
) -> list[_Candidate]:
    found: list[tuple[str, str, int]] = []
    if node.for_each is not None:
        item = infer_reference_schema(spec, capabilities, node, "$item")
        found.append(("$item", "current item", _ITEM))
        found.extend(_fields("$item", "current item", item, _ITEM))
    for other, rank in _eligible(spec, node):
        output = _output_schema(other, capabilities)
        if other.for_each is None:
            base = f"$nodes.{other.id}.output"
            found.append((base, f"{other.id} → output", rank))
            found.extend(_fields(base, other.id, output, rank))
        else:
            base = f"$nodes.{other.id}.items[*].output"
            found.append((base, f"{other.id} → output (all items)", rank))
            found.extend(_fields(base, other.id, output, rank, " (all items)"))
            found.append((f"$nodes.{other.id}.items", f"{other.id} → items (input and output of each)", rank))
    found.extend((f"$inputs.{name}", f"run input {name}", _INPUT) for name in spec.inputs)
    return sorted(
        (
            _Candidate(expr, label, rank, infer_reference_schema(spec, capabilities, node, expr))
            for expr, label, rank in found
        ),
        key=lambda candidate: candidate.rank,
    )


def _for_each_candidates(
    spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract], node: WorkflowNode
) -> list[dict[str, Any]]:
    found: list[tuple[int, str, str, str]] = []
    for name in spec.inputs:
        schema = infer_reference_schema(spec, capabilities, node, f"$inputs.{name}")
        if _is_array(schema):
            found.append((_INPUT, f"$inputs.{name}", f"run input {name}", _type_name(_items(schema))))
    for other, rank in _eligible(spec, node):
        output = _output_schema(other, capabilities)
        lists = [(name, sub) for name, sub in _properties(output).items() if _is_array(sub)]
        if other.for_each is None:
            found.extend(
                (rank, f"$nodes.{other.id}.output.{name}", f"{other.id} → each of {name}", _type_name(_items(sub)))
                for name, sub in lists
            )
            continue
        found.append((rank, f"$nodes.{other.id}.items", f"{other.id} → each item (input and output)", "object"))
        found.append((rank, f"$nodes.{other.id}.items[*].output", f"{other.id} → each item output", _type_name(output)))
        found.extend(
            (
                rank,
                f"flatten($nodes.{other.id}.items[*].output.{name})",
                f"{other.id} → each of {name} (all items)",
                _type_name(_items(sub)),
            )
            for name, sub in lists
        )
    found.sort(key=lambda entry: entry[0])
    return [{"expr": expr, "label": label, "item_type": item_type} for _, expr, label, item_type in found]


def _input_targets(
    node: WorkflowNode, capabilities: Mapping[str, CapabilityContract]
) -> tuple[dict[str, Any], set[str]]:
    capability = capabilities.get(node.capability or "") if node.type is NodeType.CAPABILITY else None
    if capability is None:
        return {}, set()
    required: Any = capability.input_schema.get("required", [])
    return _properties(capability.input_schema), {str(r) for r in required} if isinstance(required, list) else set()  # pyright: ignore[reportUnknownVariableType]


def suggest_bindings(
    spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract], node_id: str
) -> dict[str, Any]:
    """What each input of node_id can be bound to, compatible and closest first. KeyError for an unknown node."""
    node = spec.node(node_id)
    candidates = _candidates(spec, capabilities, node)
    targets, required = _input_targets(node, capabilities)
    inputs: dict[str, Any] = {}
    for name, raw_target in targets.items():
        target: dict[str, Any] = raw_target if isinstance(raw_target, dict) else {}  # pyright: ignore[reportUnknownVariableType]
        scored = sorted(
            ((candidate, is_compatible(candidate.schema, target)) for candidate in candidates),
            key=lambda pair: (not pair[1], pair[0].rank),
        )
        inputs[name] = {
            "required": name in required,
            "schema": target,
            "description": str(target.get("description") or ""),
            "candidates": [
                {"expr": c.expr, "label": c.label, "type": _type_name(c.schema), "compatible": ok} for c, ok in scored
            ],
        }
    item = [{"expr": c.expr, "label": c.label, "type": _type_name(c.schema)} for c in candidates if c.rank == _ITEM]
    return {"inputs": inputs, "for_each": _for_each_candidates(spec, capabilities, node), "item": item}


def _source_node(expr: str) -> str | None:
    try:
        ref = parse(expr)
    except ExpressionError:
        return None
    if isinstance(ref, Ref) and ref.root == "nodes" and ref.path and isinstance(ref.path[0], str):
        return ref.path[0]
    return None


def auto_wire(spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract], node_id: str) -> WorkflowSpec:
    """Copy of spec where each unbound required input of node_id with exactly one certainly-compatible candidate
    is bound to it (and its node added to depends_on). KeyError for an unknown node."""
    wired = spec.model_copy(deep=True)
    node = wired.node(node_id)
    candidates = _candidates(spec, capabilities, spec.node(node_id))
    targets, required = _input_targets(node, capabilities)
    for name, target in targets.items():
        if name not in required or name in node.inputs:
            continue
        fits = [c for c in candidates if is_compatible(c.schema, target, strict=True)]
        if len(fits) != 1:
            continue
        node.inputs[name] = fits[0].expr
        source = _source_node(fits[0].expr)
        if source is not None and source not in node.depends_on:
            node.depends_on.append(source)
    return wired
