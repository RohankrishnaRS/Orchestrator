"""WorkflowSpec validation: every structural, reference and policy rule is enforced in code before a run starts."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from ueoa.contracts import CapabilityContract, Effect, InputSpec, NodeType, WorkflowNode, WorkflowSpec

from .expressions import WILDCARD, Call, ExpressionError, Node, Ref, is_expression, parse, references
from .graph import CycleError, ancestors, node_expressions, topological_order

DEFAULT_MAX_NODES = 25

# Marks the run-context wrappers ({output} / {items, failed} / {input, output}) whose keys are fixed.
_SHAPE = "x-ueoa-shape"
_SHAPE_HINT = "for_each nodes expose items[*].input/output and failed[*].index/input/error; other nodes expose output"
_EMBEDDED = re.compile(r"\$(?:inputs|nodes|run)\.|\$item\b")
_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
_CANDIDATE_DEPTH = 3


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    node_id: str | None = None

    def __str__(self) -> str:
        where = f"[{self.node_id}] " if self.node_id else ""
        return f"{self.code}: {where}{self.message}"


def _schema_properties(schema: Mapping[str, Any]) -> tuple[set[str], set[str], bool]:
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    closed = schema.get("additionalProperties", True) is False
    return properties, required, closed


# ---------------------------------------------------------------- output shapes and paths


def _shape(properties: dict[str, Any]) -> dict[str, Any]:
    return {_SHAPE: True, "type": "object", "properties": properties}


def _list_of(item: dict[str, Any]) -> dict[str, Any]:
    return {_SHAPE: True, "type": "array", "items": item}


def _node_shape(
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
    node_id: str,
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Schema of what `$nodes.<id>` exposes at run time; None for an unknown node."""
    try:
        node = spec.node(node_id)
    except KeyError:
        return None
    if node.type is NodeType.HUMAN_INPUT:
        output: dict[str, Any] = node.input_schema or {}
    else:
        capability = capabilities.get(node.capability or "")
        output = capability.output_schema if capability is not None else {}
    if node.for_each is None:
        return _shape({"output": output})
    element = None if node_id in seen else _item_schema(node, spec, capabilities, seen | {node_id})
    item = _shape({"input": element or {}, "output": output})
    failed = _shape({"index": {"type": "integer"}, "input": {}, "error": {"type": "string"}})
    return _shape({"items": _list_of(item), "failed": _list_of(failed)})


def _segment_text(segment: str | int) -> str:
    if segment == WILDCARD:
        return "[*]"
    return f"[{segment}]" if isinstance(segment, int) else segment


def _walk_schema(
    schema: Any, path: tuple[str | int, ...]
) -> tuple[dict[str, Any] | None, tuple[str, str, list[str]] | None]:
    """Follow path through a schema -> (sub-schema or None when unknown/open, (code, segment, available) problem)."""
    current: Any = schema
    for segment in path:
        if not isinstance(current, dict):
            return None, None
        shaped = bool(current.get(_SHAPE))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        properties: Any = current.get("properties")  # pyright: ignore[reportUnknownMemberType]
        if isinstance(segment, int) or segment == WILDCARD:
            items: Any = current.get("items")  # pyright: ignore[reportUnknownMemberType]
            if isinstance(items, dict):
                current = items
                continue
            if shaped:
                return None, ("bad_shape", _segment_text(segment), sorted(properties or {}))
            return None, None
        if shaped:
            if segment not in (properties or {}):
                return None, ("bad_shape", segment, sorted(properties or {}))
        else:
            if not isinstance(properties, dict) or not properties:
                return None, None
            if segment not in properties:
                extra: Any = current.get("additionalProperties", False)  # pyright: ignore[reportUnknownMemberType]
                if extra is True or isinstance(extra, dict):
                    return None, None
                return None, ("unknown_output_field", segment, sorted(properties))  # pyright: ignore[reportUnknownArgumentType]
        current = properties[segment]
    return (current if isinstance(current, dict) else None), None  # pyright: ignore[reportUnknownVariableType]


def _path_issue(
    ref: Ref, schema: dict[str, Any] | None, path: tuple[str | int, ...], location: str, node_id: str | None
) -> Issue | None:
    if schema is None:
        return None
    _, problem = _walk_schema(schema, path)
    if problem is None:
        return None
    code, segment, available = problem
    listed = ", ".join(available) if available else "nothing (it is a list: use [*] or [n])"
    message = f"{location}: {ref.text()} - '{segment}' does not exist; available: {listed}"
    if code == "bad_shape":
        message += f" ({_SHAPE_HINT})"
    return Issue(code, message, node_id)


# ---------------------------------------------------------------- binding compatibility


@dataclass(frozen=True)
class BindingFix:
    """An incompatible input binding with exactly one compatible path under the same source."""

    node_id: str
    input: str
    source: str
    replacement: str


@dataclass(frozen=True)
class ItemInputMisuse:
    """`$item.input[...]` where the fan-out node's result fits the target but nothing declares the element does."""

    node_id: str
    input: str
    source: str
    replacement: str
    fan_out: str
    run_input: str | None  # set when the fan-out node iterates `$inputs.<run_input>`
    path: tuple[str | int, ...]
    required: tuple[str, ...]


@dataclass(frozen=True)
class _Problem:
    input: str
    ref: Ref
    reason: str
    candidates: list[str]
    misuse: ItemInputMisuse | None = None
    certain: bool = True
    hint: str = ""


def _resolve(schema: Any, path: tuple[str | int, ...]) -> dict[str, Any] | None:
    """Schema of the value at path; [*] wraps the rest in an array. None when unknown."""
    if not isinstance(schema, dict):
        return None
    if not path:
        return schema  # pyright: ignore[reportUnknownVariableType]
    head, rest = path[0], path[1:]
    if isinstance(head, int) or head == WILDCARD:
        items: Any = schema.get("items")  # pyright: ignore[reportUnknownMemberType]
        if head != WILDCARD:
            return _resolve(items, rest)
        if not isinstance(items, dict):
            return None
        inner = _resolve(items, rest)
        return {"type": "array", "items": inner} if inner is not None else {"type": "array"}
    properties: Any = schema.get("properties")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(properties, dict) or head not in properties:
        return None
    return _resolve(properties[head], rest)


def _declared_schema(declared: InputSpec) -> dict[str, Any] | None:
    items: Any = (declared.model_extra or {}).get("items")
    if declared.type in _JSON_TYPES:
        if declared.type == "array" and isinstance(items, dict):
            return {"type": "array", "items": items}
        return {"type": declared.type}
    if declared.type == "string_list":
        return {"type": "array", "items": {"type": "string"}}
    if declared.type == "list" or declared.type.endswith("_list"):
        return {"type": "array", "items": items} if isinstance(items, dict) else {"type": "array"}
    return None


def _elements(
    source: Node,
    node: WorkflowNode,
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Schema of one element of the list source produces at run time; None when not certain."""
    if isinstance(source, Call) and source.name == "flatten":
        inner = _elements(source.arg, node, spec, capabilities, seen)
        types = _types(inner) if inner is not None else None
        if types == {"array"}:
            items: Any = inner.get("items") if inner is not None else None
            return items if isinstance(items, dict) else None  # pyright: ignore[reportUnknownVariableType]
        return inner if types and "array" not in types else None
    if not isinstance(source, Ref) or source.root not in {"nodes", "inputs"}:
        return None
    listed = _ref_schema(source, node, spec, capabilities, seen)
    element: Any = listed.get("items") if listed is not None else None
    return element if isinstance(element, dict) else None  # pyright: ignore[reportUnknownVariableType]


def _item_schema(
    node: WorkflowNode,
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Schema of `$item` inside a for_each node; None when the for_each form is not inferred."""
    if node.for_each is None:
        return None
    try:
        source = parse(node.for_each)
    except ExpressionError:
        return None
    return _elements(source, node, spec, capabilities, seen)


def _ref_schema(
    ref: Ref,
    node: WorkflowNode,
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    head = ref.path[0] if ref.path else None
    if ref.root == "nodes" and isinstance(head, str):
        return _resolve(_node_shape(spec, capabilities, head, seen), ref.path[1:])
    if ref.root == "inputs" and isinstance(head, str) and head in spec.inputs:
        return _resolve(_declared_schema(spec.inputs[head]), ref.path[1:])
    if ref.root == "item":
        return _resolve(_item_schema(node, spec, capabilities, seen), ref.path)
    return None


def _types(schema: Mapping[str, Any]) -> set[str] | None:
    declared = schema.get("type")
    if isinstance(declared, str):
        types = {declared}
    elif isinstance(declared, list) and declared and all(isinstance(t, str) for t in declared):  # pyright: ignore[reportUnknownVariableType]
        types = set(declared)  # pyright: ignore[reportUnknownArgumentType]
    elif isinstance(schema.get("properties"), dict):
        types = {"object"}
    elif isinstance(schema.get("items"), dict):
        types = {"array"}
    else:
        return None
    if types & {"integer", "number"}:
        types |= {"integer", "number"}
    return types  # pyright: ignore[reportUnknownVariableType]


def _type_text(types: set[str]) -> str:
    shown = sorted(types - {"integer"}) if {"integer", "number"} <= types else sorted(types)
    return " or ".join(shown)


def _required(schema: Mapping[str, Any]) -> tuple[str, ...]:
    required: Any = schema.get("required")
    return tuple(r for r in required if isinstance(r, str)) if isinstance(required, list) else ()  # pyright: ignore[reportUnknownVariableType]


def _fit(target: Any, source: Any, label: str) -> tuple[str, str]:
    """('ok' | 'unknown' | 'bad', reason) for binding a value of schema source to schema target."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return "unknown", ""
    wanted, given = _types(target), _types(source)  # pyright: ignore[reportUnknownArgumentType]
    if wanted is None or given is None:
        return "unknown", ""
    if not wanted & given:
        return "bad", f"expects {_type_text(wanted)}; {label} is {_type_text(given)}"
    needed = _required(target)  # pyright: ignore[reportUnknownArgumentType]
    if not needed or "object" not in wanted or "object" not in given:
        return "ok", ""
    properties: Any = source.get("properties")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(properties, dict) or not properties:
        return "unknown", ""
    missing = [name for name in needed if name not in properties]  # pyright: ignore[reportUnknownVariableType]
    if not missing:
        return "ok", ""
    extra: Any = source.get("additionalProperties", False)  # pyright: ignore[reportUnknownMemberType]
    if not source.get(_SHAPE) and (extra is True or isinstance(extra, dict)):  # pyright: ignore[reportUnknownMemberType]
        return "unknown", ""
    return "bad", f"expects an object with {', '.join(needed)}; {label} has {', '.join(sorted(properties))}"  # pyright: ignore[reportUnknownArgumentType]


def infer_reference_schema(
    spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract], node: WorkflowNode, expr: str
) -> dict[str, Any] | None:
    """Schema of what a reference or flatten(...) yields inside `node`; None when it cannot be inferred."""
    try:
        source = parse(expr)
    except ExpressionError:
        return None
    if isinstance(source, Ref):
        return _ref_schema(source, node, spec, capabilities)
    if isinstance(source, Call) and source.name == "flatten":
        element = _elements(source, node, spec, capabilities)
        return {"type": "array", "items": element} if element is not None else {"type": "array"}
    return None


def is_compatible(source_schema: Any, target_schema: Any, *, strict: bool = False) -> bool:
    """Whether a value of source_schema may bind to target_schema (as incompatible_binding judges it).

    strict: only when the fit is certain (both schemas known and matching).
    """
    verdict, _ = _fit(target_schema, source_schema, "")
    return verdict == "ok" if strict else verdict != "bad"


def _sub_paths(schema: Any, depth: int) -> Iterator[tuple[str | int, ...]]:
    if depth <= 0 or not isinstance(schema, dict):
        return
    properties: Any = schema.get("properties")  # pyright: ignore[reportUnknownMemberType]
    if isinstance(properties, dict):
        for name, sub in properties.items():  # pyright: ignore[reportUnknownVariableType]
            yield (str(name),)  # pyright: ignore[reportUnknownArgumentType]
            yield from ((str(name), *rest) for rest in _sub_paths(sub, depth - 1))  # pyright: ignore[reportUnknownArgumentType]
    items: Any = schema.get("items")  # pyright: ignore[reportUnknownMemberType]
    if isinstance(items, dict):
        yield from ((WILDCARD, *rest) for rest in _sub_paths(items, depth - 1))


def _for_each_ref(node: WorkflowNode) -> Ref | None:
    if node.for_each is None:
        return None
    try:
        source = parse(node.for_each)
    except ExpressionError:
        return None
    if isinstance(source, Call) and source.name == "flatten":
        source = source.arg
    return source if isinstance(source, Ref) else None


def _fan_out_source(node: WorkflowNode) -> str | None:
    """Id of X when node iterates the {input, output} items of for_each node X."""
    source = _for_each_ref(node)
    if source is None or source.root != "nodes" or len(source.path) < 2:
        return None
    head, field, *rest = source.path
    if not isinstance(head, str) or field != "items" or any(segment != WILDCARD for segment in rest):
        return None
    return head


def _run_input(spec: WorkflowSpec, node_id: str) -> str | None:
    try:
        source = _for_each_ref(spec.node(node_id))
    except KeyError:
        return None
    if source is None or source.root != "inputs" or len(source.path) != 1 or not isinstance(source.path[0], str):
        return None
    return source.path[0]


def _item_input_misuse(
    name: str,
    ref: Ref,
    target: dict[str, Any],
    node: WorkflowNode,
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
) -> ItemInputMisuse | None:
    if ref.root != "item" or not ref.path or ref.path[0] != "input":
        return None
    fan_out = _fan_out_source(node)
    required = _required(target)
    if fan_out is None or not required or "object" not in (_types(target) or set()):
        return None
    replacement = Ref("item", ("output", *ref.path[1:]))
    if _fit(target, _ref_schema(replacement, node, spec, capabilities), "")[0] != "ok":
        return None
    return ItemInputMisuse(
        node.id, name, ref.text(), replacement.text(), fan_out, _run_input(spec, fan_out), ref.path[1:], required
    )


def _binding_problems(
    node: WorkflowNode, spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract]
) -> Iterator[_Problem]:
    """Each incompatible (or, for $item.input misuse, unproven) direct binding with compatible alternatives."""
    capability = capabilities.get(node.capability or "")
    if node.type is NodeType.HUMAN_INPUT or capability is None:
        return
    targets: Any = capability.input_schema.get("properties")
    if not isinstance(targets, dict):
        return
    for name, value in node.inputs.items():
        target: Any = targets.get(name)  # pyright: ignore[reportUnknownMemberType]
        if not is_expression(value) or not isinstance(target, dict):
            continue
        try:
            ref = parse(value)
        except ExpressionError:
            continue
        if not isinstance(ref, Ref):
            continue  # functions, defaults and operators are not inferred
        source = _ref_schema(ref, node, spec, capabilities)
        verdict, reason = _fit(target, source, ref.text())
        misuse = None if verdict == "ok" else _item_input_misuse(name, ref, target, node, spec, capabilities)  # pyright: ignore[reportUnknownArgumentType]
        if misuse is not None:
            if verdict != "bad":
                reason = (
                    f"expects an object with {', '.join(misuse.required)}; {ref.text()} is not declared to have them"
                )
            yield _Problem(name, ref, reason, [misuse.replacement], misuse, certain=verdict == "bad")
            continue
        if verdict != "bad":
            continue
        candidates = [
            Ref(ref.root, (*ref.path, *path)).text()
            for path in _sub_paths(source, _CANDIDATE_DEPTH)
            if _fit(target, _resolve(source, path), "")[0] == "ok"
        ]
        yield _Problem(name, ref, reason, candidates, hint=_list_hint(ref, source, target, node))


def _list_hint(ref: Ref, source: Any, target: Any, node: WorkflowNode) -> str:
    """How to bind a list to an input that takes one element."""
    given, wanted = _types(source or {}), _types(target)
    if given != {"array"} or wanted is None or "array" in wanted:
        return ""
    if ref.root == "item" and node.for_each is not None:
        return f' ($item is itself a list here: fan out over "flatten({node.for_each})" instead)'
    if node.for_each is None:
        return (
            f' (it takes ONE element: to call it once per element set "for_each": "{ref.text()}" and bind "$item", '
            f'or bind one element such as "{ref.text()}[0]")'
        )
    return ""


def _path_ok(ref: Ref, node: WorkflowNode, spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract]) -> bool:
    if ref.root == "item":
        return _path_issue(ref, _item_schema(node, spec, capabilities), ref.path, "", node.id) is None
    if ref.root == "nodes" and ref.path and isinstance(ref.path[0], str):
        shape = _node_shape(spec, capabilities, ref.path[0])
        return _path_issue(ref, shape, ref.path[1:], "", node.id) is None
    return True


@dataclass(frozen=True)
class PathFix:
    """A reference whose path does not exist, with exactly one wrapper slip-corrected path that does."""

    node_id: str | None  # None for workflow outputs
    location: str
    source: str
    replacement: str


def _path_candidates(ref: Ref) -> list[Ref]:
    """Paths with the run-context wrapper (output / items[*].output) added, removed or swapped."""
    path = ref.path
    if ref.root == "item":
        return [Ref("item", path[1:])] if path[:1] == ("output",) else [Ref("item", ("output", *path))]
    if ref.root != "nodes" or not path or not isinstance(path[0], str):
        return []
    head, rest = path[0], path[1:]
    fanned = ("items", WILDCARD, "output")
    if rest[:1] == ("output",):
        return [Ref("nodes", (head, *fanned, *rest[1:]))]
    if rest[:3] == fanned:
        return [Ref("nodes", (head, "output", *rest[3:]))]
    if rest and rest[0] not in {"items", "failed"}:
        return [Ref("nodes", (head, "output", *rest)), Ref("nodes", (head, *fanned, *rest))]
    return []


def _ref_ok(
    ref: Ref, node: WorkflowNode | None, spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract]
) -> bool:
    if node is not None:
        return _path_ok(ref, node, spec, capabilities)
    if ref.root == "nodes" and ref.path and isinstance(ref.path[0], str):
        shape = _node_shape(spec, capabilities, ref.path[0])
        return _path_issue(ref, shape, ref.path[1:], "", None) is None
    return True


def path_fixes(spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract]) -> list[PathFix]:
    """References with a path issue that exactly one wrapper-corrected path resolves (see _path_candidates)."""
    known = {n.id for n in spec.nodes}
    sources: list[tuple[WorkflowNode | None, str, str]] = [
        (node, e.location, e.source) for node in spec.nodes for e in node_expressions(node)
    ]
    sources += [(None, f"outputs.{name}", source) for name, source in spec.outputs.items()]
    fixes: list[PathFix] = []
    for node, location, source in sources:
        try:
            uses = references(parse(source))
        except ExpressionError:
            continue
        for use in uses:
            ref = use.ref
            if ref.root == "nodes" and (not ref.path or ref.path[0] not in known):
                continue
            if ref.root not in {"nodes", "item"} or _ref_ok(ref, node, spec, capabilities):
                continue
            valid = [c for c in _path_candidates(ref) if _ref_ok(c, node, spec, capabilities)]
            if len(valid) == 1:
                fixes.append(PathFix(node.id if node else None, location, ref.text(), valid[0].text()))
    return fixes


def binding_fixes(spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract]) -> list[BindingFix]:
    """Incompatible bindings (certain source and target schemas) with exactly one valid compatible path below."""
    fixes: list[BindingFix] = []
    for node in spec.nodes:
        for problem in _binding_problems(node, spec, capabilities):
            if not problem.certain or len(problem.candidates) != 1:
                continue
            replacement = parse(problem.candidates[0])
            if isinstance(replacement, Ref) and _path_ok(replacement, node, spec, capabilities):
                fixes.append(BindingFix(node.id, problem.input, problem.ref.text(), problem.candidates[0]))
    return fixes


def item_input_misuses(spec: WorkflowSpec, capabilities: Mapping[str, CapabilityContract]) -> list[ItemInputMisuse]:
    """`$item.input` bindings that only the real run inputs can confirm or refute (see ItemInputMisuse)."""
    return [
        problem.misuse
        for node in spec.nodes
        for problem in _binding_problems(node, spec, capabilities)
        if problem.misuse is not None and not problem.certain
    ]


def _embedded_references(value: Any, location: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        if not is_expression(value) and _EMBEDDED.search(value):
            yield location, value
    elif isinstance(value, dict):
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            yield from _embedded_references(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            yield from _embedded_references(item, f"{location}[{index}]")


# ---------------------------------------------------------------- rules


def _check_reference(
    ref: Ref,
    node: WorkflowNode,
    spec: WorkflowSpec,
    node_ancestors: set[str],
    location: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if ref.root == "nodes":
        if not ref.path or not isinstance(ref.path[0], str):
            return [Issue("bad_reference", f"{location}: {ref.text()} must name a node", node.id)]
        target = ref.path[0]
        if target not in {n.id for n in spec.nodes}:
            issues.append(Issue("unknown_node_reference", f"{location}: {ref.text()} - no node '{target}'", node.id))
        elif target not in node_ancestors:
            issues.append(
                Issue(
                    "missing_dependency",
                    f"{location}: {ref.text()} references '{target}' which is not a (transitive) dependency",
                    node.id,
                )
            )
    elif ref.root == "inputs":
        if not ref.path or ref.path[0] not in spec.inputs:
            issues.append(Issue("unknown_input", f"{location}: {ref.text()} is not a declared input", node.id))
    elif ref.root == "item" and location == "condition":
        issues.append(
            Issue("item_in_condition", f"condition: {ref.text()} - conditions apply to the whole node", node.id)
        )
    elif ref.root == "item" and node.for_each is None and location != "for_each":
        issues.append(Issue("item_without_for_each", f"{location}: {ref.text()} used outside for_each", node.id))
    elif ref.root == "item" and location == "for_each":
        issues.append(Issue("bad_reference", "for_each cannot reference $item", node.id))
    return issues


def validate_spec(
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> list[Issue]:
    issues: list[Issue] = []
    ids = [node.id for node in spec.nodes]
    known = set(ids)

    duplicates = {node_id for node_id in ids if ids.count(node_id) > 1}
    issues.extend(Issue("duplicate_node", f"node id '{d}' used more than once", d) for d in sorted(duplicates))
    if len(ids) > max_nodes:
        issues.append(Issue("too_many_nodes", f"{len(ids)} nodes exceeds the limit of {max_nodes}"))

    for node in spec.nodes:
        for dep in node.depends_on:
            if dep == node.id:
                issues.append(Issue("self_dependency", "node depends on itself", node.id))
            elif dep not in known:
                issues.append(Issue("unknown_dependency", f"depends on unknown node '{dep}'", node.id))

    try:
        topological_order(spec)
    except CycleError as exc:
        issues.append(Issue("cycle", str(exc)))
        return issues  # reference checks need an acyclic graph

    node_ancestors = ancestors(spec)
    for node in spec.nodes:
        issues.extend(_validate_node(node, spec, capabilities, node_ancestors[node.id]))

    for name, source in spec.outputs.items():
        try:
            uses = references(parse(source))
        except ExpressionError as exc:
            issues.append(Issue("bad_expression", f"outputs.{name} = {_shown(source)}: {exc}"))
            continue
        for use in uses:
            if use.ref.root == "nodes" and (not use.ref.path or use.ref.path[0] not in known):
                issues.append(Issue("unknown_node_reference", f"outputs.{name}: {use.ref.text()}"))
            elif use.ref.root not in {"nodes", "inputs", "run"}:
                issues.append(Issue("bad_reference", f"outputs.{name}: ${use.ref.root} not allowed in outputs"))
            elif use.ref.root == "nodes":
                shape = _node_shape(spec, capabilities, str(use.ref.path[0]))
                issue = _path_issue(use.ref, shape, use.ref.path[1:], f"outputs.{name}", None)
                if issue is not None:
                    issues.append(issue)
    return issues


def _validate_node(
    node: WorkflowNode,
    spec: WorkflowSpec,
    capabilities: Mapping[str, CapabilityContract],
    node_ancestors: set[str],
) -> list[Issue]:
    issues: list[Issue] = []
    item_schema = _item_schema(node, spec, capabilities)

    for expression in node_expressions(node):
        try:
            uses = references(parse(expression.source))
        except ExpressionError as exc:
            issues.append(
                Issue("bad_expression", f"{expression.location} = {_shown(expression.source)}: {exc}", node.id)
            )
            continue
        for use in uses:
            ref_issues = _check_reference(use.ref, node, spec, node_ancestors, expression.location)
            issues.extend(ref_issues)
            if ref_issues:
                continue
            if use.ref.root == "nodes":
                shape = _node_shape(spec, capabilities, str(use.ref.path[0]))
                issue = _path_issue(use.ref, shape, use.ref.path[1:], expression.location, node.id)
            elif use.ref.root == "item":
                issue = _path_issue(use.ref, item_schema, use.ref.path, expression.location, node.id)
            else:
                issue = None
            if issue is not None:
                issues.append(issue)

    for location, value in _embedded_references(node.inputs, "inputs"):
        issues.append(
            Issue(
                "embedded_reference",
                f"{location} = {_shown(value)}: expressions must start with $ and be the whole value (no references "
                'inside text); bind each value on its own, e.g. {"a": "$nodes.<id>.output.<field>", "b": "$item"}',
                node.id,
            )
        )

    if node.for_each is not None and not (
        is_expression(node.for_each) or node.for_each.lstrip().startswith("flatten(")
    ):
        issues.append(Issue("bad_for_each", "for_each must be a reference expression or flatten(...)", node.id))
    if node.max_concurrency is not None and node.for_each is None:
        issues.append(Issue("concurrency_without_for_each", "max_concurrency needs for_each", node.id))
    if node.on_item_error != "fail" and node.for_each is None:
        issues.append(Issue("on_item_error_without_for_each", "on_item_error needs for_each", node.id))

    if node.type is NodeType.HUMAN_INPUT:
        return issues

    if node.timeout_s is None:
        issues.append(Issue("missing_timeout", "capability nodes must declare timeout_s", node.id))

    capability = capabilities.get(node.capability or "")
    if capability is None:
        issues.append(Issue("unknown_capability", f"capability '{node.capability}' is not registered", node.id))
        return issues

    if capability.effects is Effect.WRITE and node.effect is not Effect.WRITE:
        issues.append(Issue("effect_mismatch", "capability writes; node must declare effect: write", node.id))
    if capability.effects is Effect.READ and node.effect is Effect.WRITE:
        issues.append(Issue("effect_mismatch", "node declares write on a read-only capability", node.id))

    if node.effect is Effect.WRITE:
        if node.approval is None or not node.approval.required:
            issues.append(Issue("write_without_approval", "write nodes require approval", node.id))
        if not node.idempotency_key:
            issues.append(Issue("write_without_idempotency", "write nodes require idempotency_key", node.id))
        if node.retry.max_attempts > 1:
            issues.append(Issue("write_with_retries", "write nodes must not retry automatically", node.id))

    properties, required, closed = _schema_properties(capability.input_schema)
    provided = set(node.inputs)
    missing = sorted(required - provided)
    if missing:
        issues.append(Issue("missing_input", f"required inputs not bound: {', '.join(missing)}", node.id))
    if closed:
        unknown = sorted(provided - properties)
        if unknown:
            issues.append(Issue("unknown_input_binding", f"inputs not in contract: {', '.join(unknown)}", node.id))
    for problem in _binding_problems(node, spec, capabilities):
        candidates = problem.candidates
        hint = ""
        if len(candidates) == 1:
            hint = f" - did you mean {candidates[0]}?"
        elif candidates:
            hint = f" - did you mean one of {', '.join(candidates[:3])}?"
        if problem.misuse is not None:
            hint += f" ($item.input is the original element given to {problem.misuse.fan_out}, not its result)"
        issues.append(
            Issue("incompatible_binding", f"inputs.{problem.input} {problem.reason}{hint}{problem.hint}", node.id)
        )
    return issues


def _shown(source: str, limit: int = 160) -> str:
    text = source if len(source) <= limit else f"{source[:limit]}..."
    return f'"{text}"'
