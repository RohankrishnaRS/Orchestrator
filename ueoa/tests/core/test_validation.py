import copy
from typing import Any

import pytest

from ueoa.contracts import CapabilityContract, WorkflowSpec
from ueoa.orchestrator.validation import BindingFix, ItemInputMisuse, binding_fixes, item_input_misuses, validate_spec


def _codes(spec: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> set[str]:
    return {issue.code for issue in validate_spec(spec, capabilities)}


def _mutate(spec: WorkflowSpec, node_id: str, **changes: Any) -> WorkflowSpec:
    raw = spec.model_dump(mode="json", exclude_none=True)
    for node in raw["nodes"]:
        if node["id"] == node_id:
            node.update(changes)
    return WorkflowSpec.model_validate(raw)


def test_pipeline_is_valid(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    assert validate_spec(pipeline, capabilities) == []


def test_cycle_detected(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    assert "cycle" in _codes(_mutate(pipeline, "ingest", depends_on=["summarize"]), capabilities)


def test_unknown_capability(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    assert "unknown_capability" in _codes(_mutate(pipeline, "score", capability="result.unknown"), capabilities)


def test_write_requires_approval_idempotency_and_no_retry(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    spec = _mutate(pipeline, "apply", approval={"required": False}, idempotency_key=None, retry={"max_attempts": 3})
    raw = spec.model_dump(mode="json", exclude_none=True)
    spec = WorkflowSpec.model_validate(raw)
    codes = _codes(spec, capabilities)
    assert {"write_without_approval", "write_without_idempotency", "write_with_retries"} <= codes


def test_write_capability_must_be_declared_write(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    assert "effect_mismatch" in _codes(_mutate(pipeline, "apply", effect="read"), capabilities)


def test_read_capability_cannot_write(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    assert "effect_mismatch" in _codes(_mutate(pipeline, "score", effect="write"), capabilities)


def test_missing_timeout(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    raw = pipeline.model_dump(mode="json", exclude_none=True)
    for node in raw["nodes"]:
        if node["id"] == "check":
            node.pop("timeout_s")
    assert "missing_timeout" in _codes(WorkflowSpec.model_validate(raw), capabilities)


def test_reference_to_non_ancestor(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    spec = _mutate(pipeline, "score", inputs={"results": "$nodes.summarize.output", "profile": "$inputs.profile"})
    assert "missing_dependency" in _codes(spec, capabilities)


def test_item_outside_for_each(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    spec = _mutate(pipeline, "score", inputs={"results": "$item.output", "profile": "$inputs.profile"})
    assert "item_without_for_each" in _codes(spec, capabilities)


def test_unknown_input_and_bindings(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    spec = _mutate(pipeline, "score", inputs={"results": "$inputs.nope", "extra": 1})
    codes = _codes(spec, capabilities)
    assert {"unknown_input", "missing_input", "unknown_input_binding"} <= codes


def test_bad_expression(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    assert "bad_expression" in _codes(_mutate(pipeline, "contrast", condition="len($inputs.x"), capabilities)


def test_human_input_requires_schema() -> None:
    with pytest.raises(ValueError, match="input_schema"):
        WorkflowSpec.model_validate(
            {"id": "w", "version": "1.0.0", "goal": "g", "nodes": [{"id": "s", "type": "human_input"}]}
        )


def _issues(spec: WorkflowSpec, capabilities: dict[str, CapabilityContract], code: str) -> list[str]:
    return [str(issue) for issue in validate_spec(spec, capabilities) if issue.code == code]


@pytest.mark.parametrize(
    ("node_id", "inputs"),
    [
        ("check", {"results": "$nodes.analyze.output.results", "items": "$nodes.ingest.items[*].output"}),
        ("score", {"results": "$nodes.check.items[*].output", "profile": "$inputs.profile"}),
        ("check", {"results": "$nodes.analyze.items.output", "items": "$nodes.ingest.items[*].output"}),
        ("check", {"results": "$nodes.analyze.failed[*].output", "items": "$nodes.ingest.items[*].output"}),
        ("analyze", {"item": "$item.result", "profile": "$inputs.profile"}),
    ],
)
def test_bad_shape(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract], node_id: str, inputs: dict[str, str]
) -> None:
    issues = _issues(_mutate(pipeline, node_id, inputs=inputs), capabilities, "bad_shape")
    assert len(issues) == 1 and "for_each nodes expose items" in issues[0]


@pytest.mark.parametrize(
    ("node_id", "inputs", "available"),
    [
        ("score", {"results": "$nodes.check.output.result", "profile": "$inputs.profile"}, "results"),
        ("check", {"results": "$nodes.analyze.items[*].output.res", "items": []}, "results"),
        ("analyze", {"item": "$item.output.name", "profile": "$inputs.profile"}, "id"),
        ("plan", {"results": [], "selected": "$nodes.choose.output.selected"}, "selected_ids"),
    ],
)
def test_unknown_output_field(
    pipeline: WorkflowSpec,
    capabilities: dict[str, CapabilityContract],
    node_id: str,
    inputs: dict[str, str],
    available: str,
) -> None:
    issues = _issues(_mutate(pipeline, node_id, inputs=inputs), capabilities, "unknown_output_field")
    assert len(issues) == 1 and issues[0].endswith(f"does not exist; available: {available}")


def test_open_output_schemas_are_not_checked(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    spec = _mutate(pipeline, "summarize", inputs={"content": "$nodes.merge.output.anything", "locale": "en"})
    assert validate_spec(spec, capabilities) == []
    opened = {**capabilities}
    check = capabilities["result.check"]
    opened["result.check"] = check.model_copy(
        update={"output_schema": {**check.output_schema, "additionalProperties": True}}
    )
    spec = _mutate(pipeline, "score", inputs={"results": "$nodes.check.output.extra", "profile": "p"})
    assert validate_spec(spec, opened) == []


def test_item_fields_checked_against_list_item_schema(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    check = capabilities["result.check"]
    typed = {**capabilities}
    typed["result.check"] = check.model_copy(
        update={
            "output_schema": {
                "type": "object",
                "properties": {"results": {"type": "array", "items": {"properties": {"id": {"type": "string"}}}}},
            }
        }
    )
    raw = pipeline.model_dump(mode="json", exclude_none=True)
    raw["nodes"].append(
        {
            "id": "per_result",
            "capability": "item.ingest",
            "depends_on": ["check"],
            "for_each": "$nodes.check.output.results",
            "inputs": {"item": {"id": "$item.id", "name": "$item.name"}},
            "timeout_s": 5,
        }
    )
    issues = _issues(WorkflowSpec.model_validate(raw), typed, "unknown_output_field")
    assert issues == [
        "unknown_output_field: [per_result] inputs.item.name: $item.name - 'name' does not exist; available: id"
    ]


def test_outputs_are_shape_and_path_checked(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    raw = pipeline.model_dump(mode="json", exclude_none=True)
    raw["outputs"] = {"summary": "$nodes.summarize.output.txt", "applied": "$nodes.apply.output"}
    codes = [issue.code for issue in validate_spec(WorkflowSpec.model_validate(raw), capabilities)]
    assert sorted(codes) == ["bad_shape", "unknown_output_field"]


def test_embedded_reference(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    spec = _mutate(pipeline, "score", inputs={"results": "$nodes.check.output.results", "profile": "p-$inputs.x"})
    [issue] = _issues(spec, capabilities, "embedded_reference")
    assert issue.startswith(
        'embedded_reference: [score] inputs.profile = "p-$inputs.x": expressions must start with $ and be the whole '
        "value (no references inside text)"
    )
    spec = _mutate(pipeline, "apply", inputs={"payload": {"note": ["see $item"]}})
    assert "embedded_reference" in _codes(spec, capabilities)


def test_item_in_condition(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    spec = _mutate(pipeline, "apply", condition="$item.key == 'x'")
    assert "item_in_condition" in _codes(spec, capabilities)


def test_swarm_bounds_and_on_item_error(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    with pytest.raises(ValueError, match="less than or equal to 16"):
        _mutate(pipeline, "ingest", max_concurrency=17)
    assert "on_item_error_without_for_each" in _codes(
        _mutate(pipeline, "score", on_item_error="continue"), capabilities
    )
    assert validate_spec(_mutate(pipeline, "analyze", on_item_error="continue"), capabilities) == []


def test_references_inside_flatten_are_checked(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    raw = pipeline.model_dump(mode="json", exclude_none=True)
    node = {
        "id": "per_result",
        "capability": "item.ingest",
        "for_each": "flatten($nodes.analyze.items[*].output.results)",
        "inputs": {"item": {"id": "$item"}},
        "timeout_s": 5,
    }
    raw["nodes"].append(node)
    assert "missing_dependency" in _codes(WorkflowSpec.model_validate(raw), capabilities)
    node["depends_on"] = ["analyze"]
    assert validate_spec(WorkflowSpec.model_validate(raw), capabilities) == []
    node["for_each"] = "flatten($nodes.analyze.items[*].output.result)"
    assert "unknown_output_field" in _codes(WorkflowSpec.model_validate(raw), capabilities)


def _record_input(capabilities: dict[str, CapabilityContract]) -> dict[str, CapabilityContract]:
    analyze = capabilities["item.analyze"]
    schema = copy.deepcopy(analyze.input_schema)
    schema["properties"]["item"] = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}
    return {**capabilities, "item.analyze": analyze.model_copy(update={"input_schema": schema})}


def test_item_wrapper_bound_where_output_is_expected(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    typed = _record_input(capabilities)
    assert validate_spec(pipeline, typed) == []
    spec = _mutate(pipeline, "analyze", inputs={"item": "$item", "profile": "$inputs.profile"})
    assert _issues(spec, typed, "incompatible_binding") == [
        "incompatible_binding: [analyze] inputs.item expects an object with id; $item has input, output"
        " - did you mean $item.output?"
    ]
    assert binding_fixes(spec, typed) == [BindingFix("analyze", "item", "$item", "$item.output")]


@pytest.mark.parametrize(
    ("node_id", "inputs", "message"),
    [
        (
            "score",
            {"results": "$nodes.check.output", "profile": "$inputs.profile"},
            "inputs.results expects array; $nodes.check.output is object - did you mean $nodes.check.output.results?",
        ),
        (
            "score",
            {"results": "$nodes.check.output.results", "profile": "$inputs.items"},
            "inputs.profile expects string; $inputs.items is array",
        ),
        (
            "summarize",
            {"content": "$nodes.analyze.items[*].output", "locale": "en"},
            "inputs.content expects object; $nodes.analyze.items[*].output is array",
        ),
    ],
)
def test_type_mismatched_binding(
    pipeline: WorkflowSpec,
    capabilities: dict[str, CapabilityContract],
    node_id: str,
    inputs: dict[str, str],
    message: str,
) -> None:
    issues = _issues(_mutate(pipeline, node_id, inputs=inputs), capabilities, "incompatible_binding")
    assert len(issues) == 1 and issues[0].startswith(f"incompatible_binding: [{node_id}] {message}"), issues


@pytest.mark.parametrize(
    "binding",
    ["$nodes.check.output | default(null)", "flatten($nodes.analyze.items[*].output.results)", "$run.locale"],
)
def test_non_reference_or_unknown_sources_are_not_shape_checked(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract], binding: str
) -> None:
    spec = _mutate(pipeline, "merge", inputs={"results": binding, "flags": "$nodes.score.output.flags"})
    assert _issues(spec, capabilities, "incompatible_binding") == []


def test_open_source_object_is_not_reported_missing_fields(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    typed = _record_input(capabilities)
    ingest = typed["item.ingest"]
    typed["item.ingest"] = ingest.model_copy(
        update={"output_schema": {"type": "object", "properties": {"ref": {}}, "additionalProperties": True}}
    )
    assert validate_spec(pipeline, typed) == []
    typed["item.ingest"] = ingest.model_copy(
        update={"output_schema": {"type": "object", "properties": {"ref": {"type": "string"}}}}
    )
    issues = _issues(pipeline, typed, "incompatible_binding")
    assert issues == ["incompatible_binding: [analyze] inputs.item expects an object with id; $item.output has ref"]
    assert binding_fixes(pipeline, typed) == []


def _with_parts(capabilities: dict[str, CapabilityContract]) -> dict[str, CapabilityContract]:
    typed = _record_input(capabilities)
    ingest = typed["item.ingest"]
    record = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}
    schema = {
        **ingest.output_schema,
        "properties": {"id": {"type": "string"}, "parts": {"type": "array", "items": record}},
    }
    return {**typed, "item.ingest": ingest.model_copy(update={"output_schema": schema})}


@pytest.mark.parametrize(
    "for_each",
    [
        "$nodes.ingest.items[*].output",
        "$nodes.ingest.items[*].output.parts[0]",
        "flatten($nodes.ingest.items[*].output)",
        "flatten($nodes.ingest.items[*].output.parts)",
    ],
)
def test_item_is_the_element_when_for_each_maps_or_flattens(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract], for_each: str
) -> None:
    typed = _with_parts(capabilities)
    spec = _mutate(pipeline, "analyze", for_each=for_each, inputs={"item": "$item", "profile": "$inputs.profile"})
    assert validate_spec(spec, typed) == []
    assert binding_fixes(spec, typed) == []

    wrapped = _mutate(spec, "analyze", inputs={"item": "$item.output", "profile": "$inputs.profile"})
    issues = _issues(wrapped, typed, "unknown_output_field")
    assert len(issues) == 1 and "$item.output - 'output' does not exist" in issues[0]
    assert binding_fixes(wrapped, typed) == []


@pytest.mark.parametrize("for_each", ["$nodes.ingest.items", "$nodes.ingest.items[*]", "flatten($nodes.ingest.items)"])
def test_item_wrapper_sources_still_rebind_to_output(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract], for_each: str
) -> None:
    typed = _record_input(capabilities)
    spec = _mutate(pipeline, "analyze", for_each=for_each, inputs={"item": "$item", "profile": "$inputs.profile"})
    assert binding_fixes(spec, typed) == [BindingFix("analyze", "item", "$item", "$item.output")]
    fixed = _mutate(spec, "analyze", inputs={"item": "$item.output", "profile": "$inputs.profile"})
    assert validate_spec(fixed, typed) == []


def test_item_is_list_element_of_a_node_output_field(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    typed = _with_parts(capabilities)
    spec = _mutate(pipeline, "analyze", for_each="$nodes.ingest.items[0].output.parts", inputs={"item": "$item.nope"})
    issues = _issues(spec, typed, "unknown_output_field")
    assert issues == ["unknown_output_field: [analyze] inputs.item: $item.nope - 'nope' does not exist; available: id"]


@pytest.mark.parametrize(
    "for_each",
    ["$inputs.items", "flatten([$nodes.ingest.items[*].output])", "$nodes.ingest.items[*].output | default([])"],
)
@pytest.mark.parametrize("binding", ["$item", "$item.output", "$item.anything"])
def test_unknown_for_each_sources_are_never_checked_or_rebound(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract], for_each: str, binding: str
) -> None:
    typed = _record_input(capabilities)
    spec = _mutate(pipeline, "analyze", for_each=for_each, inputs={"item": binding, "profile": "$inputs.profile"})
    codes = _codes(spec, typed)
    assert not codes & {"incompatible_binding", "unknown_output_field", "bad_shape"}, codes
    assert binding_fixes(spec, typed) == []


def test_declared_input_item_schema_types_item(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    typed = _record_input(capabilities)
    raw = pipeline.model_dump(mode="json", exclude_none=True)
    raw["inputs"]["items"] = {"type": "list", "items": {"type": "object", "properties": {"id": {"type": "string"}}}}
    spec = _mutate(
        WorkflowSpec.model_validate(raw), "analyze", for_each="$inputs.items", inputs={"item": "$item.output"}
    )
    issues = _issues(spec, typed, "unknown_output_field")
    assert issues == [
        "unknown_output_field: [analyze] inputs.item: $item.output - 'output' does not exist; available: id"
    ]


def test_item_input_of_fan_out_result_is_flagged_not_auto_rebound(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    typed = _record_input(capabilities)
    spec = _mutate(pipeline, "analyze", inputs={"item": "$item.input", "profile": "$inputs.profile"})
    assert _issues(spec, typed, "incompatible_binding") == [
        "incompatible_binding: [analyze] inputs.item expects an object with id; $item.input is not declared to have"
        " them - did you mean $item.output? ($item.input is the original element given to ingest, not its result)"
    ]
    assert binding_fixes(spec, typed) == []
    assert item_input_misuses(spec, typed) == [
        ItemInputMisuse("analyze", "item", "$item.input", "$item.output", "ingest", "items", (), ("id",))
    ]


def test_item_input_typed_from_the_fan_out_source(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    typed = _record_input(capabilities)
    raw = pipeline.model_dump(mode="json", exclude_none=True)
    record = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}
    raw["inputs"]["items"] = {"type": "list", "items": record}
    declared = _mutate(
        WorkflowSpec.model_validate(raw), "analyze", inputs={"item": "$item.input", "profile": "$inputs.profile"}
    )
    assert validate_spec(declared, typed) == [] and item_input_misuses(declared, typed) == []

    raw["inputs"]["items"] = {"type": "list", "items": {"type": "object", "properties": {"ref": {"type": "string"}}}}
    lacking = _mutate(
        WorkflowSpec.model_validate(raw), "analyze", inputs={"item": "$item.input", "profile": "$inputs.profile"}
    )
    assert _issues(lacking, typed, "incompatible_binding") == [
        "incompatible_binding: [analyze] inputs.item expects an object with id; $item.input has ref"
        " - did you mean $item.output? ($item.input is the original element given to ingest, not its result)"
    ]
    assert binding_fixes(lacking, typed) == [BindingFix("analyze", "item", "$item.input", "$item.output")]
    assert item_input_misuses(lacking, typed) == []


def test_fan_out_input_inference_survives_for_each_cycles(capabilities: dict[str, CapabilityContract]) -> None:
    typed = _record_input(capabilities)
    node = {"capability": "item.analyze", "inputs": {"item": "$item.input", "profile": "p"}, "timeout_s": 5}
    spec = WorkflowSpec.model_validate(
        {
            "id": "loop",
            "version": "1.0.0",
            "goal": "g",
            "nodes": [
                {**node, "id": "a", "for_each": "$nodes.b.items"},
                {**node, "id": "b", "for_each": "$nodes.a.items"},
            ],
        }
    )
    assert binding_fixes(spec, typed) == [] and item_input_misuses(spec, typed) == []
    assert "missing_dependency" in _codes(spec, typed)
