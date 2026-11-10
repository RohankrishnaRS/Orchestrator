"""Run controller: executes workflows through a gateway, concurrently, with retries, skips and human input."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from ueoa.api.app import Services, Settings, build_services
from ueoa.contracts import CapabilityContract, WorkflowSpec
from ueoa.gateway.service import GatewayDeniedError, GatewayFailedError, InvocationRecord, Principal, Step
from ueoa.orchestrator.runner import RunController, RunError
from ueoa.orchestrator.scheduler import NodeState, RunStatus
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa_sdk.host import InvocationContext, create_agent_app

pytestmark = pytest.mark.anyio

USER = Principal("tester", "user")
Behaviour = Callable[[dict[str, Any], int], dict[str, Any]]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _record(
    capability: str, status: str, output: dict[str, Any] | None = None, error: str | None = None
) -> InvocationRecord:
    return InvocationRecord(
        invocation_id=str(uuid.uuid4()),
        at="2026-01-01T00:00:00+00:00",
        principal=USER,
        capability=capability,
        status=status,
        implementation=f"fake-{capability}@1.0.0",
        output=output,
        error=error,
    )


def fail(capability: str, error: str, code: str | None = None) -> GatewayFailedError:
    exc = GatewayFailedError(_record(capability, "failed", error=error))
    exc.code = code  # type: ignore[assignment]  # None exercises the message-prefix fallback
    return exc


def deny(capability: str, step: str, reason: str) -> GatewayDeniedError:
    record = _record(capability, "denied", error=reason)
    record.steps.append(Step(step, "deny", reason))
    return GatewayDeniedError(record)


def _flatten(lists: list[list[Any]]) -> list[Any]:
    return [value for inner in lists for value in inner]


PIPELINE_BEHAVIOUR: dict[str, Behaviour] = {
    "item.ingest": lambda p, _n: {"id": p["item"]["id"]},
    "item.analyze": lambda p, _n: {"results": [p["item"]["id"]]},
    "result.check": lambda p, _n: {"results": _flatten(p["results"])},
    "result.score": lambda p, _n: {"flags": []},
    "result.contrast": lambda p, _n: {"rows": [{"key": k} for k in p["keys"]]},
    "result.merge": lambda p, _n: {"results": p["results"], "comparison": p.get("comparison")},
    "text.summarize": lambda p, _n: {"text": f"{len(p['content']['results'])} results in {p['locale']}"},
    "action.plan": lambda p, _n: {"proposals": [{"key": s, "payload": {"id": s}} for s in p["selected"]]},
}


class FakeGateway:
    """Deterministic gateway double: per-capability behaviour, call log and concurrency high-water marks."""

    def __init__(self, behaviour: dict[str, Behaviour], delay: float = 0.01) -> None:
        self.behaviour = behaviour
        self.delay = delay
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active: dict[str, int] = defaultdict(int)
        self.peak: dict[str, int] = defaultdict(int)
        self.total_active = 0
        self.total_peak = 0

    def count(self, capability: str) -> int:
        return sum(1 for name, _ in self.calls if name == capability)

    async def invoke(self, principal: Principal, capability_id: str, payload: dict[str, Any]) -> InvocationRecord:
        self.calls.append((capability_id, payload))
        attempt = self.count(capability_id)
        self.active[capability_id] += 1
        self.peak[capability_id] = max(self.peak[capability_id], self.active[capability_id])
        self.total_active += 1
        self.total_peak = max(self.total_peak, self.total_active)
        try:
            await asyncio.sleep(self.delay)
            behaviour = self.behaviour.get(capability_id)
            if behaviour is None:
                raise deny(capability_id, "effect", "write capability needs approval")
            output = behaviour(payload, attempt)
        finally:
            self.active[capability_id] -= 1
            self.total_active -= 1
        return _record(capability_id, "succeeded", output)


def _items(count: int) -> list[dict[str, str]]:
    return [{"id": f"i{n}"} for n in range(count)]


def _controller(gateway: Any, capabilities: dict[str, CapabilityContract], **options: Any) -> RunController:
    return RunController(gateway, lambda: capabilities, max_backoff_s=0, **options)


async def test_pipeline_fans_out_concurrently_waits_for_input_and_completes(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    gateway = FakeGateway(PIPELINE_BEHAVIOUR)
    controller = _controller(gateway, capabilities)

    run = await controller.execute(pipeline, {"items": _items(10)}, USER, plan={"rationale": "test"})

    assert run.status is RunStatus.WAITING
    assert run.nodes["choose"].state is NodeState.WAITING_INPUT
    assert (gateway.peak["item.ingest"], gateway.peak["item.analyze"]) == (6, 4)
    assert gateway.count("item.ingest") == gateway.count("item.analyze") == 10
    assert [item.output for item in run.nodes["ingest"].items] == [{"id": f"i{n}"} for n in range(10)]
    assert run.nodes["contrast"].state is NodeState.SKIPPED
    merge_payload = next(p for name, p in gateway.calls if name == "result.merge")
    assert merge_payload["comparison"] is None
    assert run.outputs is None and run.finished_at is None

    with pytest.raises(RunError, match="input rejected"):
        await controller.submit_input(run.id, "choose", {"selected_ids": "not-a-list"})
    with pytest.raises(RunError, match="not waiting"):
        await controller.submit_input(run.id, "merge", {"selected_ids": []})

    await controller.submit_input(run.id, "choose", {"selected_ids": []})
    run = await controller.wait(run.id, timeout=5)

    assert run.status is RunStatus.COMPLETED
    assert run.nodes["plan"].state is NodeState.SKIPPED
    assert run.nodes["apply"].state is NodeState.SKIPPED
    assert run.outputs == {"summary": {"text": "10 results in en"}, "applied": []}
    assert run.events[0].type == "run_started" and run.events[-1].type == "run_finished"
    assert {"node_waiting_input", "input_received", "item_started", "item_succeeded"} <= {e.type for e in run.events}
    view = json.loads(json.dumps(run.view()))
    assert view["plan"] == {"rationale": "test"} and view["nodes"]["choose"]["state"] == "succeeded"
    assert view["workflow"]["id"] == pipeline.id and view["goal"] == pipeline.goal
    assert run.summary()["nodes"] == {"succeeded": 7, "skipped": 3}
    assert controller.list() == [run] and controller.get(run.id) is run


async def test_contrast_runs_when_condition_true(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    gateway = FakeGateway(PIPELINE_BEHAVIOUR)
    run = await _controller(gateway, capabilities).execute(pipeline, {"items": _items(2), "contrast_keys": ["k"]}, USER)

    assert run.nodes["contrast"].state is NodeState.SUCCEEDED
    merge_payload = next(p for name, p in gateway.calls if name == "result.merge")
    assert merge_payload["comparison"] == {"rows": [{"key": "k"}]}


async def test_write_node_waits_for_approval_and_reject_completes_rejected(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    gateway = FakeGateway(PIPELINE_BEHAVIOUR)
    controller = _controller(gateway, capabilities)
    run = await controller.execute(pipeline, {"items": _items(1)}, USER)
    await controller.submit_input(run.id, "choose", {"selected_ids": ["i0"]})
    run = await controller.wait(run.id)

    assert run.status is RunStatus.WAITING
    apply = run.nodes["apply"]
    assert (apply.state, apply.items[0].state) == (NodeState.WAITING_APPROVAL, NodeState.WAITING_APPROVAL)
    [pending] = run.view()["approvals"]
    assert (pending["node"], pending["item"], pending["payload"]) == ("apply", 0, {"payload": {"id": "i0"}})
    assert pending["idempotency_key"] == f"{run.id}:apply:i0:{pending['payload_hash']}"
    assert gateway.count("action.apply") == 0

    reviewer = Principal("reviewer", "user")
    await controller.decide(
        run.id, "apply", item=0, decision="reject", principal=reviewer, payload_hash=pending["payload_hash"]
    )
    run = await controller.wait(run.id, timeout=5)

    assert run.status is RunStatus.COMPLETED_REJECTED
    assert (apply.state, apply.items[0].error) == (NodeState.REJECTED, "rejected by reviewer")
    assert gateway.count("action.apply") == 0 and run.approvals == {}


async def test_denial_blocks_downstream(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    def denied(_p: dict[str, Any], _n: int) -> dict[str, Any]:
        raise deny("result.check", "entitlement", "role 'user' may not use result.check")

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "result.check": denied})
    run = await _controller(gateway, capabilities).execute(pipeline, {"items": _items(2)}, USER)

    assert run.status is RunStatus.FAILED
    assert run.nodes["check"].state is NodeState.FAILED
    assert run.error == "check: denied at entitlement: role 'user' may not use result.check"
    assert all(run.nodes[n].state is NodeState.PENDING for n in ("score", "contrast", "merge", "summarize"))
    assert gateway.count("result.check") == 1
    assert run.finished_at is not None and run.outputs == {"summary": None, "applied": []}
    assert any(e.type == "output_failed" for e in run.events)


async def test_retry_on_gateway_timeout_then_success(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    def flaky(payload: dict[str, Any], attempt: int) -> dict[str, Any]:
        if attempt == 1:
            raise fail("item.analyze", "timeout: agent too slow", "timeout")
        return {"results": [payload["item"]["id"]]}

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "item.analyze": flaky})
    run = await _controller(gateway, capabilities).execute(pipeline, {"items": _items(1)}, USER)

    item = run.nodes["analyze"].items[0]
    assert (item.state, item.attempts, item.error) == (NodeState.SUCCEEDED, 2, None)
    assert item.implementation == "fake-item.analyze@1.0.0"
    assert [e.detail for e in run.events if e.type == "item_retry"] == [
        "attempt 1 failed (timeout: agent too slow); retrying in 0s"
    ]
    assert run.status is RunStatus.WAITING


async def test_retries_are_bounded_and_only_for_listed_codes(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    def down(_p: dict[str, Any], _n: int) -> dict[str, Any]:
        raise fail("item.analyze", "unavailable: connection refused", "unavailable")

    def broken(_p: dict[str, Any], _n: int) -> dict[str, Any]:
        raise fail("item.analyze", "output schema: 'results' is a required property", "bad_output")

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "item.analyze": down})
    run = await _controller(gateway, capabilities).execute(pipeline, {"items": _items(2)}, USER)
    assert [i.attempts for i in run.nodes["analyze"].items] == [3, 3]
    assert run.status is RunStatus.FAILED and run.nodes["analyze"].error == "item 0: unavailable: connection refused"

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "item.analyze": broken})
    run = await _controller(gateway, capabilities).execute(pipeline, {"items": _items(1)}, USER)
    assert run.nodes["analyze"].items[0].attempts == 1
    assert run.nodes["check"].state is NodeState.PENDING


async def test_real_timeout_counts_as_timeout(capabilities: dict[str, CapabilityContract]) -> None:
    spec = WorkflowSpec.model_validate(
        {
            "id": "slow",
            "version": "1.0.0",
            "goal": "Time out once",
            "nodes": [
                {
                    "id": "first",
                    "capability": "result.score",
                    "inputs": {"results": [], "profile": "p"},
                    "timeout_s": 1,
                    "retry": {"max_attempts": 2, "retry_on": ["timeout"]},
                }
            ],
        }
    )

    class SlowOnce(FakeGateway):
        async def invoke(self, principal: Principal, capability_id: str, payload: dict[str, Any]) -> InvocationRecord:
            if not self.calls:
                self.calls.append((capability_id, payload))
                await asyncio.sleep(5)
            return await super().invoke(principal, capability_id, payload)

    run = await _controller(SlowOnce(PIPELINE_BEHAVIOUR), capabilities).execute(spec, {}, USER)

    assert run.status is RunStatus.COMPLETED
    assert run.nodes["first"].attempts == 2
    assert run.outputs == {"first": {"flags": []}}
    assert [e.detail for e in run.events if e.type == "item_retry"] == [
        "attempt 1 failed (timeout: exceeded 1s); retrying in 0.0s"
    ]


async def test_invalid_spec_and_missing_inputs_are_rejected(
    pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]
) -> None:
    controller = _controller(FakeGateway(PIPELINE_BEHAVIOUR), capabilities)
    broken = pipeline.model_copy(deep=True)
    broken.nodes[0].capability = "item.unknown"

    with pytest.raises(RunError, match="unknown_capability"):
        await controller.start(broken, {"items": _items(1)}, USER)
    with pytest.raises(RunError, match="missing required inputs: items"):
        await controller.start(pipeline, {}, USER)
    with pytest.raises(RunError, match="unknown run"):
        controller.get("nope")
    assert controller.list() == []


async def test_empty_fan_out_and_unhandled_errors(capabilities: dict[str, CapabilityContract]) -> None:
    spec = WorkflowSpec.model_validate(
        {
            "id": "edges",
            "version": "1.0.0",
            "goal": "Edge cases",
            "inputs": {"items": {"type": "any"}},
            "nodes": [
                {
                    "id": "each",
                    "capability": "item.ingest",
                    "for_each": "$inputs.items",
                    "inputs": {"item": "$item"},
                    "timeout_s": 5,
                },
                {"id": "bad", "capability": "result.score", "inputs": {"results": [], "profile": "p"}, "timeout_s": 5},
            ],
        }
    )

    def explode(_p: dict[str, Any], _n: int) -> dict[str, Any]:
        raise KeyError("boom")

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "result.score": explode})
    run = await _controller(gateway, capabilities).execute(spec, {"items": []}, USER)

    assert run.nodes["each"].state is NodeState.SUCCEEDED and run.nodes["each"].items == []
    assert run.nodes["bad"].error == "KeyError: 'boom'"
    assert run.status is RunStatus.FAILED

    run = await _controller(gateway, capabilities).execute(spec, {"items": "nope"}, USER)
    assert run.nodes["each"].error == "for_each must produce a list, got str"


def _swarm(on_item_error: str = "continue", **ingest: Any) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "id": "swarm",
            "version": "1.0.0",
            "goal": "Fan out, tolerate item failures, join",
            "inputs": {"items": {"type": "array"}},
            "nodes": [
                {
                    "id": "each",
                    "capability": "item.ingest",
                    "for_each": "$inputs.items",
                    "on_item_error": on_item_error,
                    "inputs": {"item": "$item"},
                    "timeout_s": 5,
                    **ingest,
                },
                {
                    "id": "join",
                    "capability": "result.score",
                    "depends_on": ["each"],
                    "inputs": {"results": "$nodes.each.items[*].output", "profile": "p"},
                    "timeout_s": 5,
                },
            ],
            "outputs": {"ids": "$nodes.each.items[*].output.id", "failed": "$nodes.each.failed"},
        }
    )


def _ingest_failing(*bad: str) -> Behaviour:
    def behaviour(payload: dict[str, Any], _n: int) -> dict[str, Any]:
        if payload["item"]["id"] in bad:
            raise fail("item.ingest", "remote_error: boom", "remote_error")
        return {"id": payload["item"]["id"]}

    return behaviour


async def test_on_item_error_continue_keeps_succeeded_items(capabilities: dict[str, CapabilityContract]) -> None:
    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "item.ingest": _ingest_failing("i1")})
    run = await _controller(gateway, capabilities).execute(_swarm(), {"items": _items(3)}, USER)

    assert run.status is RunStatus.COMPLETED, run.error
    each = run.nodes["each"]
    assert each.state is NodeState.SUCCEEDED and each.error is None
    assert [item.state for item in each.items] == [NodeState.SUCCEEDED, NodeState.FAILED, NodeState.SUCCEEDED]
    assert next(p for name, p in gateway.calls if name == "result.score")["results"] == [{"id": "i0"}, {"id": "i2"}]
    assert run.outputs == {
        "ids": ["i0", "i2"],
        "failed": [{"index": 1, "input": {"id": "i1"}, "error": "remote_error: boom"}],
    }
    assert [i["state"] for i in run.view()["nodes"]["each"]["items"]] == ["succeeded", "failed", "succeeded"]
    assert any(e.type == "node_succeeded" and e.detail == "1 of 3 items failed" for e in run.events)


async def test_on_item_error_continue_fails_when_every_item_fails(
    capabilities: dict[str, CapabilityContract],
) -> None:
    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "item.ingest": _ingest_failing("i0", "i1")})
    run = await _controller(gateway, capabilities).execute(_swarm(), {"items": _items(2)}, USER)
    assert run.nodes["each"].state is NodeState.FAILED and run.status is RunStatus.FAILED

    run = await _controller(gateway, capabilities).execute(_swarm("fail"), {"items": _items(2)}, USER)
    assert run.nodes["each"].state is NodeState.FAILED

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "item.ingest": _ingest_failing("i1")})
    run = await _controller(gateway, capabilities).execute(_swarm("fail"), {"items": _items(2)}, USER)
    assert run.nodes["each"].state is NodeState.FAILED and run.nodes["each"].error == "item 1: remote_error: boom"


async def test_for_each_list_length_is_capped(capabilities: dict[str, CapabilityContract]) -> None:
    gateway = FakeGateway(PIPELINE_BEHAVIOUR)
    controller = _controller(gateway, capabilities, max_items_per_node=3)

    run = await controller.execute(_swarm(), {"items": _items(4)}, USER)

    assert run.nodes["each"].state is NodeState.FAILED
    assert run.nodes["each"].error == "for_each produced 4 items; the limit is 3 per node"
    assert gateway.calls == []


async def test_parallel_invocations_are_capped_per_run(capabilities: dict[str, CapabilityContract]) -> None:
    spec = WorkflowSpec.model_validate(
        {
            "id": "wide",
            "version": "1.0.0",
            "goal": "Two wide fan-outs in parallel",
            "inputs": {"items": {"type": "array"}},
            "nodes": [
                {
                    "id": node_id,
                    "capability": "item.ingest",
                    "for_each": "$inputs.items",
                    "max_concurrency": 16,
                    "inputs": {"item": "$item"},
                    "timeout_s": 5,
                }
                for node_id in ("left", "right")
            ],
        }
    )
    gateway = FakeGateway(PIPELINE_BEHAVIOUR)

    run = await _controller(gateway, capabilities, max_parallel_invocations=3).execute(spec, {"items": _items(6)}, USER)

    assert run.status is RunStatus.COMPLETED
    assert gateway.count("item.ingest") == 12 and gateway.total_peak == 3


def _single(retry: dict[str, Any]) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "id": "single",
            "version": "1.0.0",
            "goal": "One node",
            "nodes": [
                {
                    "id": "only",
                    "capability": "result.score",
                    "inputs": {"results": [], "profile": "p"},
                    "timeout_s": 5,
                    "retry": retry,
                }
            ],
        }
    )


@pytest.mark.parametrize("code", ["bad_output", None])
async def test_bad_output_is_retried_only_when_listed(
    capabilities: dict[str, CapabilityContract], code: str | None
) -> None:
    def bad_once(_p: dict[str, Any], attempt: int) -> dict[str, Any]:
        if attempt == 1:
            raise fail("result.score", "output schema: 'flags' is a required property", code)
        return {"flags": []}

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "result.score": bad_once})
    listed = _single({"max_attempts": 2, "retry_on": ["bad_output"]})
    run = await _controller(gateway, capabilities).execute(listed, {}, USER)
    assert run.status is RunStatus.COMPLETED and run.nodes["only"].attempts == 2

    gateway = FakeGateway({**PIPELINE_BEHAVIOUR, "result.score": bad_once})
    unlisted = _single({"max_attempts": 2, "retry_on": ["timeout", "remote_error"]})
    run = await _controller(gateway, capabilities).execute(unlisted, {}, USER)
    assert run.status is RunStatus.FAILED and run.nodes["only"].attempts == 1


async def test_run_inputs_are_type_checked(pipeline: WorkflowSpec, capabilities: dict[str, CapabilityContract]) -> None:
    controller = _controller(FakeGateway(PIPELINE_BEHAVIOUR), capabilities)

    with pytest.raises(RunError, match="invalid inputs: items: 'nope' is not of type 'array'"):
        await controller.start(pipeline, {"items": "nope"}, USER)
    with pytest.raises(RunError, match=r"invalid inputs: items: \[\] "):
        await controller.start(pipeline, {"items": []}, USER)
    with pytest.raises(RunError, match=r"invalid inputs: items: \[\{"):
        await controller.start(pipeline, {"items": _items(21)}, USER)
    with pytest.raises(RunError, match="profile: 3 is not of type 'string'; contrast_keys: 1 is not of type 'string'"):
        await controller.start(pipeline, {"items": _items(1), "contrast_keys": [1], "profile": 3}, USER)
    assert controller.list() == []


async def test_flatten_fans_out_over_nested_lists(capabilities: dict[str, CapabilityContract]) -> None:
    spec = WorkflowSpec.model_validate(
        {
            "id": "nested",
            "version": "1.0.0",
            "goal": "Fan out over every result of every item",
            "inputs": {"items": {"type": "array"}},
            "nodes": [
                {
                    "id": "analyze",
                    "capability": "item.analyze",
                    "for_each": "$inputs.items",
                    "inputs": {"item": "$item", "profile": "p"},
                    "timeout_s": 5,
                },
                {
                    "id": "each_result",
                    "capability": "item.ingest",
                    "depends_on": ["analyze"],
                    "for_each": "flatten($nodes.analyze.items[*].output.results)",
                    "inputs": {"item": {"id": "$item"}},
                    "timeout_s": 5,
                },
            ],
            "outputs": {"ids": "$nodes.each_result.items[*].output.id"},
        }
    )

    run = await _controller(FakeGateway(PIPELINE_BEHAVIOUR), capabilities).execute(spec, {"items": _items(3)}, USER)

    assert run.status is RunStatus.COMPLETED, run.error
    assert run.outputs == {"ids": ["i0", "i1", "i2"]}


# ---------------------------------------------------------------- real registry + gateway + SDK agent

TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}


def _contract(capability_id: str, description: str) -> dict[str, Any]:
    return {
        "id": capability_id,
        "version": "1.0.0",
        "description": description,
        "effects": "read",
        "input_schema": TEXT_SCHEMA,
        "output_schema": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
        "samples": [{"text": "hello"}],
    }


TEXT_CARD: dict[str, Any] = {
    "name": "Text Agent",
    "description": "Echoes and upper-cases text.",
    "version": "1.0.0",
    "capabilities": {
        "extensions": [
            {
                "uri": UEOA_EXTENSION_URI,
                "params": {
                    "id": "text-agent",
                    "owner": "platform-tests",
                    "kind": "agent",
                    "implementation": "deterministic",
                    "binding": "a2a",
                    "implements": ["text.echo@1", "text.upper@1"],
                    "effects": "read",
                    "timeout_s": 5,
                },
            }
        ]
    },
    "skills": [
        {"id": "echo", "name": "Echo", "description": "Echo text"},
        {"id": "upper", "name": "Upper", "description": "Upper-case text"},
    ],
}


async def _echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


async def _upper(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"].upper()}


@pytest.fixture
async def services() -> AsyncIterator[Services]:
    agent = create_agent_app(TEXT_CARD, {"text.echo": _echo, "text.upper": _upper}, "http://text.test")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=agent)) as http:
        yield await build_services(Settings(catalog_dir=None, allowed_hosts=("text.test",)), http)


async def test_runs_through_real_registry_gateway_and_agent(services: Services) -> None:
    services.registry.register_capability(_contract("text.echo", "Return the input text unchanged."))
    services.registry.register_capability(_contract("text.upper", "Return the input text in upper case."))
    impl = await services.registry.register_from_url("http://text.test")
    assert impl.conformant, [c for c in impl.checks if not c.passed]
    services.registry.approve(impl.key)
    spec = WorkflowSpec.model_validate(
        {
            "id": "two-step",
            "version": "1.0.0",
            "goal": "Echo then upper-case each word",
            "inputs": {"words": {"type": "string_list"}},
            "nodes": [
                {"id": "first", "capability": "text.echo", "inputs": {"text": "start"}, "timeout_s": 5},
                {
                    "id": "second",
                    "capability": "text.upper",
                    "depends_on": ["first"],
                    "for_each": "$inputs.words",
                    "inputs": {"text": "$item"},
                    "timeout_s": 5,
                },
            ],
            "outputs": {"echo": "$nodes.first.output.text", "words": "$nodes.second.items[*].output.text"},
        }
    )
    controller = RunController(services.gateway, lambda: services.registry.capabilities)

    run = await controller.execute(spec, {"words": ["a", "b", "c"]}, USER)

    assert run.status is RunStatus.COMPLETED, run.error
    assert run.outputs == {"echo": "start", "words": ["A", "B", "C"]}
    assert run.nodes["first"].implementation == "text-agent@1.0.0"
    audited = {record.invocation_id for record in services.gateway.audit}
    assert run.nodes["first"].invocation_id in audited
    assert {item.invocation_id for item in run.nodes["second"].items} <= audited
