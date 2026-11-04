"""LLM planner against the neutral fixture catalog with a scripted model double."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from ueoa.contracts import CapabilityContract, WorkflowSpec
from ueoa.llm import LLMError
from ueoa.orchestrator import planner as planner_module
from ueoa.orchestrator.planner import SYSTEM_PROMPT, Planner, describe_inputs, rank_capabilities

pytestmark = pytest.mark.anyio

GOAL = "Ingest each item, analyze it against the profile and check the results"
INPUTS: dict[str, Any] = {"items": [{"id": "a", "body": "one"}, {"id": "b", "body": "two"}], "profile": "standard"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedLLM:
    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        self._replies = list(replies)
        self.systems: list[str] = []
        self.prompts: list[str] = []
        self.chain_ids: list[str | None] = []

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        self.systems.append(system)
        self.prompts.append(user)
        self.chain_ids.append(chain_id)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return copy.deepcopy(reply)


def _workflow() -> dict[str, Any]:
    return {
        "id": "ingest-analyze",
        "version": "1.0.0",
        "goal": GOAL,
        "inputs": {"items": {"type": "array"}, "profile": {"type": "string"}},
        "nodes": [
            {
                "id": "ingest",
                "capability": "item.ingest",
                "for_each": "$inputs.items",
                "max_concurrency": 4,
                "inputs": {"item": "$item"},
                "timeout_s": 60,
            },
            {
                "id": "analyze",
                "capability": "item.analyze",
                "depends_on": ["ingest"],
                "for_each": "$nodes.ingest.items",
                "inputs": {"item": "$item.output", "profile": "$inputs.profile"},
                "timeout_s": 60,
            },
            {
                "id": "check",
                "capability": "result.check",
                "depends_on": ["analyze"],
                "inputs": {
                    "results": "$nodes.analyze.items[*].output.results",
                    "items": "$nodes.ingest.items[*].output",
                },
                "timeout_s": 60,
            },
        ],
        "outputs": {"checked": "$nodes.check.output.results"},
    }


def _reply(workflow: dict[str, Any]) -> dict[str, Any]:
    return {"rationale": "fan out per item, then join", "workflow": workflow}


async def test_valid_fan_out_plan_is_accepted(capabilities: dict[str, CapabilityContract]) -> None:
    llm = ScriptedLLM(_reply(_workflow()))
    result = await Planner(llm).plan(GOAL, INPUTS, capabilities, chain_id="chain-1")

    assert result.ok, result.issues
    assert result.attempts == 1 and llm.chain_ids == ["chain-1"]
    assert result.spec is not None and result.spec.node("ingest").for_each == "$inputs.items"
    assert result.rationale == "fan out per item, then join"
    view = result.view()
    assert json.loads(json.dumps(view)) == view
    assert view["ok"] is True and view["workflow"]["id"] == "ingest-analyze"
    assert llm.systems == [SYSTEM_PROMPT] and "untrusted" in SYSTEM_PROMPT


async def test_auto_fixes_make_almost_valid_plan_pass_without_repair(
    capabilities: dict[str, CapabilityContract],
) -> None:
    workflow = _workflow()
    del workflow["id"], workflow["version"], workflow["goal"], workflow["inputs"]
    for node in workflow["nodes"]:
        del node["timeout_s"]
    workflow["nodes"][2]["depends_on"] = []
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm, default_timeout_s=90).plan(GOAL, INPUTS, capabilities)

    assert result.ok, result.issues
    assert len(llm.prompts) == 1
    spec = result.spec
    assert spec is not None
    assert spec.id.startswith("plan-") and spec.version == "1.0.0" and spec.goal == GOAL
    assert {name: s.type for name, s in spec.inputs.items()} == {"items": "array", "profile": "string"}
    assert all(node.timeout_s == 90 for node in spec.nodes)
    assert set(spec.node("check").depends_on) == {"analyze", "ingest"}


async def test_repair_loop_sends_back_issues(capabilities: dict[str, CapabilityContract]) -> None:
    broken = _workflow()
    broken["nodes"][0]["inputs"] = {}
    llm = ScriptedLLM(_reply(broken), _reply(_workflow()))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert result.ok and result.attempts == 2
    assert "required inputs not bound: item" in llm.prompts[1]
    assert "YOUR PREVIOUS ANSWER" in llm.prompts[1] and '"inputs":{}' in llm.prompts[1]


async def test_run_input_shape_mismatch_goes_to_repair(capabilities: dict[str, CapabilityContract]) -> None:
    # binds a string from the real run inputs where the capability needs an object
    scalar = _workflow()
    scalar["nodes"][0]["inputs"] = {"item": "$item.body"}
    llm = ScriptedLLM(_reply(scalar), _reply(_workflow()))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert result.ok and result.attempts == 2
    assert "run_input_mismatch: [ingest] inputs.item = $item.body gives str" in llm.prompts[1]


async def test_direct_run_input_binding_is_checked_against_real_values(
    capabilities: dict[str, CapabilityContract],
) -> None:
    direct = _workflow()
    direct["nodes"][0] = {"id": "ingest", "capability": "item.ingest", "inputs": {"item": "$inputs.profile"}}
    direct["nodes"][1]["for_each"] = "$inputs.items"
    direct["nodes"][1]["inputs"]["item"] = "$item"
    llm = ScriptedLLM(_reply(direct), _reply(direct), _reply(direct))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert not result.ok
    assert any(i.startswith("run_input_mismatch: [ingest] inputs.item = $inputs.profile") for i in result.issues)
    assert not any("[analyze]" in i and "run_input_mismatch" in i for i in result.issues)  # elements are objects


async def test_gives_up_after_max_attempts(capabilities: dict[str, CapabilityContract]) -> None:
    bad = _workflow()
    bad["nodes"][2]["capability"] = "action.apply"
    llm = ScriptedLLM(_reply(bad), {"rationale": "no workflow"})

    result = await Planner(llm, max_attempts=2).plan(GOAL, INPUTS, capabilities)

    assert not result.ok and result.attempts == 2 and len(llm.prompts) == 2
    assert result.spec is None and result.issues == ['answer must contain a "workflow" object (or "cannot_plan": true)']
    assert "unknown_capability" in llm.prompts[1] and "action.apply" in llm.prompts[1]


async def test_pydantic_errors_are_reported(capabilities: dict[str, CapabilityContract]) -> None:
    bad = _workflow()
    bad["nodes"][0]["timeout_s"] = -1
    llm = ScriptedLLM(_reply(bad))

    result = await Planner(llm, max_attempts=1).plan(GOAL, INPUTS, capabilities)

    assert not result.ok and result.spec is None
    assert any(issue.startswith("schema: nodes.0.timeout_s") for issue in result.issues)


async def test_auto_fix_slugifies_ids_and_drops_unknown_keys(capabilities: dict[str, CapabilityContract]) -> None:
    workflow = _workflow()
    workflow["id"] = "Ingest Analyze"
    workflow["notes"] = "free text"
    ingest, analyze, check = workflow["nodes"]
    ingest["id"], ingest["label"] = "ingestItems", "x"
    analyze["id"], analyze["depends_on"] = "analyze-each", ["ingestItems"]
    analyze["for_each"] = "$nodes.ingestItems.items"
    check["depends_on"] = ["analyze-each"]
    check["inputs"] = {
        "results": "$nodes.analyze-each.items[*].output.results",
        "items": "$nodes.ingestItems.items[*].output",
    }
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert result.ok, result.issues
    spec = result.spec
    assert spec is not None and spec.id == "ingest-analyze"
    assert [node.id for node in spec.nodes] == ["ingest_items", "analyze_each", "check"]
    assert spec.node("analyze_each").depends_on == ["ingest_items"]
    assert spec.node("analyze_each").for_each == "$nodes.ingest_items.items"
    assert set(spec.node("check").depends_on) == {"analyze_each", "ingest_items"}
    assert spec.node("check").inputs["results"] == "$nodes.analyze_each.items[*].output.results"
    assert result.warnings == [
        "dropped unknown workflow key 'notes'",
        "dropped unknown key 'label' on node 'ingestItems'",
        "renamed node 'ingestItems' to 'ingest_items'",
        "renamed node 'analyze-each' to 'analyze_each'",
    ]
    assert result.view()["warnings"] == result.warnings


async def test_auto_fix_sets_swarm_and_retry_defaults(capabilities: dict[str, CapabilityContract]) -> None:
    workflow = _workflow()
    workflow["nodes"][0]["max_concurrency"] = 12
    workflow["nodes"][2]["retry"] = {"max_attempts": 1}
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert result.ok, result.issues
    assert result.warnings == []
    spec = result.spec
    assert spec is not None
    ingest, analyze, check = spec.nodes
    assert (ingest.on_item_error, ingest.max_concurrency) == ("continue", 8)
    assert (analyze.on_item_error, analyze.max_concurrency) == ("continue", 4)
    assert (check.on_item_error, check.max_concurrency, check.retry.max_attempts) == ("fail", None, 1)
    assert ingest.retry.model_dump() == {
        "max_attempts": 2,
        "backoff_s": [2],
        "retry_on": ["timeout", "unavailable", "rate_limited", "bad_output"],
    }


def _record_input(capabilities: dict[str, CapabilityContract]) -> dict[str, CapabilityContract]:
    analyze = capabilities["item.analyze"]
    schema = copy.deepcopy(analyze.input_schema)
    schema["properties"]["item"] = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}
    return {**capabilities, "item.analyze": analyze.model_copy(update={"input_schema": schema})}


async def test_auto_fix_rebinds_item_wrapper_to_its_output(capabilities: dict[str, CapabilityContract]) -> None:
    workflow = _workflow()
    workflow["nodes"][1]["inputs"]["item"] = "$item"
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm).plan(GOAL, INPUTS, _record_input(capabilities))

    assert result.ok, result.issues
    assert result.attempts == 1 and len(llm.prompts) == 1
    assert result.spec is not None and result.spec.node("analyze").inputs["item"] == "$item.output"
    assert result.warnings == ["rebound 'analyze' inputs.item from $item to $item.output"]


async def test_item_input_rebound_to_output_when_run_inputs_lack_the_fields(
    capabilities: dict[str, CapabilityContract],
) -> None:
    workflow = _workflow()
    workflow["nodes"][1]["inputs"]["item"] = "$item.input"
    inputs = {**INPUTS, "items": [{"body": "one"}, {"body": "two"}]}
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm).plan(GOAL, inputs, _record_input(capabilities))

    assert result.ok, result.issues
    assert result.attempts == 1 and len(llm.prompts) == 1
    assert result.spec is not None and result.spec.node("analyze").inputs["item"] == "$item.output"
    assert result.warnings == ["rebound 'analyze' inputs.item from $item.input to $item.output"]


async def test_item_input_kept_when_run_inputs_have_the_fields(capabilities: dict[str, CapabilityContract]) -> None:
    workflow = _workflow()
    workflow["nodes"][1]["inputs"]["item"] = "$item.input"
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm).plan(GOAL, INPUTS, _record_input(capabilities))

    assert result.ok, result.issues
    assert result.attempts == 1
    assert result.spec is not None and result.spec.node("analyze").inputs["item"] == "$item.input"
    assert result.warnings == ["declared inputs.items items with id (seen in the run inputs)"]
    assert result.spec.inputs["items"].model_extra == {
        "items": {"type": "object", "required": ["id"], "properties": {"id": {}}, "additionalProperties": True}
    }


async def test_item_input_without_run_inputs_goes_to_repair(capabilities: dict[str, CapabilityContract]) -> None:
    broken = _workflow()
    broken["nodes"][1]["inputs"]["item"] = "$item.input"
    llm = ScriptedLLM(_reply(broken), _reply(_workflow()))

    result = await Planner(llm).plan(GOAL, {"profile": "standard"}, _record_input(capabilities))

    assert result.ok and result.attempts == 2 and result.warnings == []
    assert "did you mean $item.output? ($item.input is the original element given to ingest" in llm.prompts[1]


async def test_invalid_spec_after_rebind_is_reported_not_raised(
    capabilities: dict[str, CapabilityContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    class SecondValidationFails(WorkflowSpec):
        @classmethod
        def model_validate(cls, obj: Any, **kwargs: Any) -> WorkflowSpec:  # type: ignore[override]
            calls.append(obj)
            return WorkflowSpec.model_validate({} if len(calls) == 2 else obj, **kwargs)

    monkeypatch.setattr(planner_module, "WorkflowSpec", SecondValidationFails)
    workflow = _workflow()
    workflow["nodes"][1]["inputs"]["item"] = "$item"

    result = await Planner(ScriptedLLM(_reply(workflow)), max_attempts=1).plan(
        GOAL, INPUTS, _record_input(capabilities)
    )

    assert len(calls) == 2 and not result.ok and result.spec is None
    assert result.issues and all(issue.startswith("schema: ") for issue in result.issues)
    assert result.warnings == ["rebound 'analyze' inputs.item from $item to $item.output"]


async def test_item_over_mapped_outputs_is_kept_as_is(capabilities: dict[str, CapabilityContract]) -> None:
    workflow = _workflow()
    workflow["nodes"][1]["for_each"] = "$nodes.ingest.items[*].output"
    workflow["nodes"][1]["inputs"]["item"] = "$item"
    llm = ScriptedLLM(_reply(workflow))

    result = await Planner(llm).plan(GOAL, INPUTS, _record_input(capabilities))

    assert result.ok, result.issues
    assert result.spec is not None and result.spec.node("analyze").inputs["item"] == "$item"
    assert result.warnings == []


async def test_wrapper_path_over_mapped_outputs_goes_to_repair(capabilities: dict[str, CapabilityContract]) -> None:
    broken = _workflow()
    broken["nodes"][1]["for_each"] = "$nodes.ingest.items[*].output"
    llm = ScriptedLLM(_reply(broken), _reply(_workflow()))

    result = await Planner(llm).plan(GOAL, INPUTS, _record_input(capabilities))

    assert result.ok and result.attempts == 2 and result.warnings == []
    assert "$item.output - 'output' does not exist; available: id" in llm.prompts[1]


async def test_unfixable_incompatible_binding_goes_to_repair(capabilities: dict[str, CapabilityContract]) -> None:
    broken = _workflow()
    broken["nodes"][2]["inputs"]["results"] = "$nodes.ingest.items[0].output"
    llm = ScriptedLLM(_reply(broken), _reply(_workflow()))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert result.ok and result.attempts == 2
    assert "incompatible_binding: [check] inputs.results expects array" in llm.prompts[1]


def test_system_prompt_documents_swarm_shape() -> None:
    for phrase in ("on_item_error", "failed (each", "flatten(x)", "never embed", "$item is not allowed in condition"):
        assert phrase in SYSTEM_PROMPT
    assert "$item.input is the original element given to the fan-out node" in SYSTEM_PROMPT


async def test_cannot_plan(capabilities: dict[str, CapabilityContract]) -> None:
    llm = ScriptedLLM({"cannot_plan": True, "rationale": "nothing fits"})

    result = await Planner(llm).plan("Do something unrelated", {}, capabilities)

    assert not result.ok and result.spec is None and result.attempts == 1
    assert result.rationale == "nothing fits" and result.issues == ["cannot_plan: nothing fits"]


async def test_only_available_read_capabilities_are_offered(capabilities: dict[str, CapabilityContract]) -> None:
    available = set(capabilities) - {"result.score"}
    llm = ScriptedLLM({"cannot_plan": True, "rationale": "needs a write"})

    result = await Planner(llm).plan("Score results and apply the action", {}, capabilities, available)

    assert "result.score" not in result.considered and "action.apply" not in result.considered
    assert "result.score" not in llm.prompts[0] and "action.apply" not in llm.prompts[0]
    assert set(result.considered) == available - {"action.apply"}
    assert "action.apply" in result.rationale


async def test_nothing_offered_skips_the_model(capabilities: dict[str, CapabilityContract]) -> None:
    llm = ScriptedLLM()

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities, available={"action.apply"})

    assert result.rationale == "no capabilities available" and result.attempts == 0 and result.issues
    assert llm.prompts == []


async def test_prompt_never_contains_full_input_values(capabilities: dict[str, CapabilityContract]) -> None:
    tail = "TAIL-MARKER"
    inputs = {"text": "x" * 5000 + tail, "items": [{"k": "v" * 500}] * 3, "count": 3}
    llm = ScriptedLLM({"cannot_plan": True, "rationale": "-"})

    await Planner(llm).plan(GOAL, inputs, capabilities)

    prompt = llm.prompts[0]
    assert tail not in prompt and "x" * 201 not in prompt and "v" * 50 not in prompt
    described = describe_inputs(inputs)
    assert described["text"]["preview"].endswith("(+4811 chars)")
    assert described["items"]["preview"] == {"length": 3, "first_item_keys": ["k"]}
    assert described["count"] == {"type": "integer", "shape": "integer", "preview": 3}


def test_rank_capabilities_is_lexical_and_deterministic(capabilities: dict[str, CapabilityContract]) -> None:
    ranked = [c.id for c in rank_capabilities("score the results against a profile", capabilities, 3)]
    assert ranked[0] == "result.score" and len(ranked) == 3
    assert ranked == [c.id for c in rank_capabilities("score the results against a profile", capabilities, 3)]
    assert [c.id for c in rank_capabilities("zzz", capabilities, 100)] == sorted(capabilities)


async def test_planner_caps_offered_capabilities(capabilities: dict[str, CapabilityContract]) -> None:
    llm = ScriptedLLM({"cannot_plan": True, "rationale": "-"})

    result = await Planner(llm, max_capabilities=2).plan("summarize text in a locale", {}, capabilities)

    assert len(result.considered) == 2 and result.considered[0] == "text.summarize"


async def test_llm_error_is_reported(capabilities: dict[str, CapabilityContract]) -> None:
    llm = ScriptedLLM(LLMError("LLM gateway returned HTTP 503"))

    result = await Planner(llm).plan(GOAL, INPUTS, capabilities)

    assert not result.ok and result.spec is None
    assert result.rationale == "planner model call failed: LLM gateway returned HTTP 503"
    assert result.issues == ["LLM gateway returned HTTP 503"]
