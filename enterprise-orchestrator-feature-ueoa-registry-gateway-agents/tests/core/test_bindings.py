"""Binding suggestions and auto-wiring over the neutral fixture catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ueoa.contracts import WorkflowSpec
from ueoa.orchestrator.bindings import auto_wire, suggest_bindings
from ueoa.orchestrator.validation import validate_spec
from ueoa.registry.seed import load_capabilities, load_workflow

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "catalog"
CAPABILITIES = load_capabilities(CATALOG / "capabilities")
PIPELINE = load_workflow(CATALOG / "workflows" / "pipeline.yaml")
DOWNSTREAM_OF_CHECK = {"score", "contrast", "merge", "summarize", "choose", "plan", "apply"}


def _by_expr(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["expr"]: c for c in candidates}


def _referenced_nodes(candidates: list[dict[str, Any]]) -> set[str]:
    return {c["expr"].split(".")[1] for c in candidates if "$nodes." in c["expr"]}


def test_downstream_of_for_each_offers_item_paths_with_compatibility() -> None:
    suggestions = suggest_bindings(PIPELINE, CAPABILITIES, "check")
    results = suggestions["inputs"]["results"]
    assert (results["required"], results["schema"]) == (True, {"type": "array"})
    found = _by_expr(results["candidates"])
    assert found["$nodes.analyze.items[*].output.results"]["compatible"] is True
    assert found["$nodes.analyze.items[*].output"] == {
        "expr": "$nodes.analyze.items[*].output",
        "label": "analyze → output (all items)",
        "type": "array",
        "compatible": True,
    }
    assert found["$nodes.analyze.items"]["compatible"] is True
    assert found["$nodes.ingest.items[*].output.id"]["label"] == "ingest → id (all items)"
    assert found["$inputs.profile"] == {
        "expr": "$inputs.profile",
        "label": "run input profile",
        "type": "string",
        "compatible": False,
    }
    flags = [c["compatible"] for c in results["candidates"]]
    assert flags == sorted(flags, reverse=True)  # compatible first
    compatible_nodes = [
        c["expr"].split(".")[1] for c in results["candidates"] if c["compatible"] and "$nodes." in c["expr"]
    ]
    assert compatible_nodes.index("analyze") < compatible_nodes.index("ingest")  # direct dependency first


def test_no_descendants_offered() -> None:
    suggestions = suggest_bindings(PIPELINE, CAPABILITIES, "check")
    offered = _referenced_nodes(suggestions["inputs"]["results"]["candidates"])
    assert offered == {"ingest", "analyze"}
    assert not offered & DOWNSTREAM_OF_CHECK
    assert not _referenced_nodes(suggestions["for_each"]) & DOWNSTREAM_OF_CHECK
    # the first node: everything else depends on it
    assert (
        _referenced_nodes(suggest_bindings(PIPELINE, CAPABILITIES, "ingest")["inputs"]["item"]["candidates"]) == set()
    )


def test_item_candidates_under_for_each() -> None:
    suggestions = suggest_bindings(PIPELINE, CAPABILITIES, "analyze")
    assert [c["expr"] for c in suggestions["item"]] == ["$item", "$item.input", "$item.output", "$item.output.id"]
    assert _by_expr(suggestions["item"])["$item.output.id"] == {
        "expr": "$item.output.id",
        "label": "current item → output.id",
        "type": "string",
    }
    candidates = suggestions["inputs"]["item"]["candidates"]
    item = _by_expr(candidates)
    assert item["$item.output"]["compatible"] is True
    assert item["$item.output.id"]["compatible"] is False
    assert [c["expr"] for c in candidates[:3]] == ["$item", "$item.input", "$item.output"]  # compatible, closest
    assert suggest_bindings(PIPELINE, CAPABILITIES, "check")["item"] == []


def test_human_input_fields_are_candidates() -> None:
    found = _by_expr(suggest_bindings(PIPELINE, CAPABILITIES, "plan")["inputs"]["selected"]["candidates"])
    assert found["$nodes.choose.output.selected_ids"] == {
        "expr": "$nodes.choose.output.selected_ids",
        "label": "choose → selected_ids",
        "type": "array",
        "compatible": True,
    }
    assert found["$nodes.summarize.output.text"]["compatible"] is False


def test_for_each_candidates() -> None:
    found = _by_expr(suggest_bindings(PIPELINE, CAPABILITIES, "apply")["for_each"])
    assert found["$nodes.plan.output.proposals"]["label"] == "plan → each of proposals"
    assert found["$nodes.ingest.items"]["item_type"] == "object"
    assert found["$nodes.ingest.items[*].output"]["item_type"] == "object"
    assert "flatten($nodes.analyze.items[*].output.results)" in found
    assert found["$inputs.contrast_keys"]["item_type"] == "string"
    assert found["$inputs.items"]["item_type"] == "unknown"
    assert "$inputs.profile" not in found
    assert "$nodes.summarize.output.text" not in found  # not a list
    assert next(iter(found)) == "$nodes.plan.output.proposals"  # direct dependency first


def _two_steps() -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "id": "two-steps",
            "version": "1.0.0",
            "goal": "Ingest then summarise",
            "inputs": {"record": {"type": "object"}},
            "nodes": [
                {"id": "a", "capability": "item.ingest", "inputs": {"item": "$inputs.record"}, "timeout_s": 30},
                {"id": "b", "capability": "text.summarize", "timeout_s": 30},
            ],
        }
    )


def test_auto_wire_binds_the_single_compatible_candidate() -> None:
    spec = _two_steps()
    wired = auto_wire(spec, CAPABILITIES, "b")
    node = wired.node("b")
    # locale has one string candidate; content has two objects (a's output and the record) so stays unbound
    assert node.inputs == {"locale": "$nodes.a.output.id"}
    assert node.depends_on == ["a"]
    assert spec.node("b").inputs == {} and spec.node("b").depends_on == []  # input not mutated
    assert [i.code for i in validate_spec(wired, CAPABILITIES)] == ["missing_input"]


def test_auto_wire_leaves_bound_inputs_and_unknown_node() -> None:
    assert auto_wire(PIPELINE, CAPABILITIES, "check") == PIPELINE
    with pytest.raises(KeyError):
        auto_wire(PIPELINE, CAPABILITIES, "nope")
    with pytest.raises(KeyError):
        suggest_bindings(PIPELINE, CAPABILITIES, "nope")
