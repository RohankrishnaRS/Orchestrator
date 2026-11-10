"""Payload-bound human approval for writes: gateway checks, run pauses and decisions, and the HTTP endpoints."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app
from ueoa.contracts import WorkflowSpec
from ueoa.gateway.invokers import Invoker
from ueoa.gateway.service import Approval, Entitlements, Gateway, GatewayDeniedError, Principal, payload_hash
from ueoa.orchestrator.runner import RunController, RunError
from ueoa.orchestrator.scheduler import NodeState, RunStatus
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa.registry.service import Registry
from ueoa_sdk.host import InvocationContext, build_card, create_agent_app

BASE = "http://agent.test"
REQUESTER = Principal("requester", "user")
REVIEWER = Principal("reviewer", "user")
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}
WRITE_NODE: dict[str, Any] = {
    "id": "write",
    "capability": "record.write",
    "effect": "write",
    "approval": {"required": True, "binds_to": "payload_hash"},
    "retry": {"max_attempts": 1},
    "timeout_s": 30,
}
SINGLE_WRITE: dict[str, Any] = {
    "id": "write-one",
    "version": "1.0.0",
    "goal": "Write one record",
    "inputs": {"text": {"type": "string"}},
    "nodes": [{**WRITE_NODE, "inputs": {"text": "$inputs.text"}, "idempotency_key": "{run.id}:{item.payload_hash}"}],
    "outputs": {"text": "$nodes.write.output.text"},
}
FAN_OUT_WRITE: dict[str, Any] = {
    "id": "write-many",
    "version": "1.0.0",
    "goal": "Write several records",
    "inputs": {"texts": {"type": "string_list"}},
    "nodes": [
        {
            **WRITE_NODE,
            "for_each": "$inputs.texts",
            "inputs": {"text": "$item"},
            "idempotency_key": "{run.id}:{node.id}:{item.index}:{item.value}",
        }
    ],
    "outputs": {"written": "$nodes.write.items[*].output | default([])"},
}


def _contract(capability_id: str, effects: str) -> dict[str, Any]:
    return {
        "id": capability_id,
        "version": "1.0.0",
        "description": f"Neutral {capability_id}.",
        "effects": effects,
        "input_schema": SCHEMA,
        "output_schema": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
        "samples": [{"text": "sample"}],
    }


def _card(agent_id: str, implements: str, effects: str) -> dict[str, Any]:
    params = {
        "id": agent_id,
        "owner": "platform-tests",
        "kind": "agent",
        "implementation": "deterministic",
        "binding": "a2a",
        "implements": [implements],
        "effects": effects,
        "timeout_s": 5,
    }
    return {
        "name": agent_id,
        "description": "Test agent.",
        "version": "1.0.0",
        "capabilities": {"extensions": [{"uri": UEOA_EXTENSION_URI, "params": params}]},
        "skills": [{"id": "s", "name": "S", "description": "s"}],
    }


class Agent:
    """SDK-hosted A2A agent serving a read and a write capability; records every write it receives."""

    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

        async def echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
            return {"text": payload["text"]}

        async def write(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
            self.writes.append(payload)
            return {"text": f"wrote {payload['text']}"}

        app = create_agent_app(_card("echo", "text.echo@1", "read"), {"text.echo": echo, "record.write": write}, BASE)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))


@dataclass
class Env:
    registry: Registry
    invoker: Invoker
    gateway: Gateway
    agent: Agent

    def gateway_with(self, **options: Any) -> Gateway:
        return Gateway(self.registry, self.invoker, Entitlements({"user": ["*"]}), **options)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    agent = Agent()
    async with agent.http:
        invoker = Invoker(agent.http)
        registry = Registry({}, invoker, agent.http, ("agent.test",))
        registry.register_capability(_contract("record.write", "write"))
        writer = await registry.register_card(build_card(BASE, _card("writer", "record.write@1", "write")))
        registry.approve(writer.key)
        assert agent.writes == []  # conformance never executes write samples
        yield Env(registry, invoker, Gateway(registry, invoker, Entitlements({"user": ["*"]})), agent)


def _approval(payload: dict[str, Any], by: str = REVIEWER.name, key: str | None = None) -> Approval:
    return Approval("a-1", payload_hash(payload), by, "2026-01-01T00:00:00+00:00", key)


# ---------------------------------------------------------------- gateway


def test_payload_hash_is_canonical() -> None:
    assert payload_hash({"b": 1, "a": "é"}) == payload_hash({"a": "é", "b": 1})
    assert payload_hash({"a": 1}) != payload_hash({"a": 2})


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("approval", "reason"),
    [
        (None, "write capability requires a human approval bound to this payload"),
        (_approval({"text": "other"}), "approval does not match this payload"),
        (_approval({"text": "x"}, by=REQUESTER.name), "four-eyes: approver must differ from requester"),
    ],
)
async def test_writes_are_denied_without_a_matching_four_eyes_approval(
    env: Env, approval: Approval | None, reason: str
) -> None:
    with pytest.raises(GatewayDeniedError) as denied:
        await env.gateway.invoke(REQUESTER, "record.write", {"text": "x"}, approval=approval)
    step = denied.value.record.steps[-1]
    assert (step.name, step.outcome, step.detail) == ("effect", "deny", reason)
    assert env.agent.writes == []


@pytest.mark.anyio
async def test_approved_write_runs_and_self_approval_needs_the_setting(env: Env) -> None:
    payload = {"text": "x"}
    record = await env.gateway.invoke(REQUESTER, "record.write", payload, approval=_approval(payload))
    assert record.output == {"text": "wrote x"}
    effect = next(step for step in record.steps if step.name == "effect")
    assert (effect.outcome, effect.detail) == ("pass", "write approved by reviewer (a-1)")

    relaxed = env.gateway_with(allow_self_approval=True)
    mine = _approval(payload, by=REQUESTER.name)
    assert (await relaxed.invoke(REQUESTER, "record.write", payload, approval=mine)).status == "succeeded"
    assert env.agent.writes == [payload, payload]


@pytest.mark.anyio
async def test_repeated_idempotency_key_replays_without_invoking(env: Env) -> None:
    payload = {"text": "once"}
    first = await env.gateway.invoke(REQUESTER, "record.write", payload, approval=_approval(payload, key="k1"))
    again = await env.gateway.invoke(REQUESTER, "record.write", payload, approval=_approval(payload, key="k1"))
    assert again.output == first.output == {"text": "wrote once"}
    assert again.steps[-1].name == "idempotent_replay" and again.implementation == first.implementation
    assert env.agent.writes == [payload]


# ---------------------------------------------------------------- run controller


def _controller(env: Env) -> RunController:
    return RunController(env.gateway, lambda: env.registry.capabilities, max_backoff_s=0)


@pytest.mark.anyio
async def test_run_pauses_for_approval_then_writes_once_approved(env: Env) -> None:
    controller = _controller(env)
    run = await controller.execute(WorkflowSpec.model_validate(SINGLE_WRITE), {"text": "x"}, REQUESTER)

    assert (run.status, run.nodes["write"].state) == (RunStatus.WAITING, NodeState.WAITING_APPROVAL)
    [pending] = run.view()["approvals"]
    assert (pending["node"], pending["item"], pending["payload"]) == ("write", None, {"text": "x"})
    assert pending["payload_hash"] == payload_hash({"text": "x"})
    assert pending["idempotency_key"] == f"{run.id}:{pending['payload_hash']}"
    assert env.agent.writes == [] and controller.pending_approvals() == [(run, run.approvals["write"])]

    with pytest.raises(RunError, match="stale approval"):
        await controller.decide(run.id, "write", item=None, decision="approve", principal=REVIEWER, payload_hash="x")
    with pytest.raises(RunError, match="four-eyes"):
        await controller.decide(
            run.id, "write", item=None, decision="approve", principal=REQUESTER, payload_hash=pending["payload_hash"]
        )
    assert run.nodes["write"].state is NodeState.WAITING_APPROVAL and "write" in run.approvals
    assert run.status is RunStatus.WAITING and env.agent.writes == []

    await controller.decide(
        run.id, "write", item=None, decision="approve", principal=REVIEWER, payload_hash=pending["payload_hash"]
    )
    run = await controller.wait(run.id, timeout=5)

    assert (run.status, run.outputs) == (RunStatus.COMPLETED, {"text": "wrote x"})
    assert run.nodes["write"].attempts == 1 and env.agent.writes == [{"text": "x"}]
    assert {"approval_requested", "approval_granted", "node_succeeded"} <= {e.type for e in run.events}
    with pytest.raises(RunError, match="not waiting for approval"):
        await controller.decide(
            run.id, "write", item=None, decision="approve", principal=REVIEWER, payload_hash=pending["payload_hash"]
        )


@pytest.mark.anyio
async def test_rejected_write_completes_rejected(env: Env) -> None:
    controller = _controller(env)
    run = await controller.execute(WorkflowSpec.model_validate(SINGLE_WRITE), {"text": "x"}, REQUESTER)
    digest = run.approvals["write"].payload_hash

    await controller.decide(run.id, "write", item=None, decision="reject", principal=REVIEWER, payload_hash=digest)
    run = await controller.wait(run.id, timeout=5)

    assert (run.status, run.nodes["write"].state) == (RunStatus.COMPLETED_REJECTED, NodeState.REJECTED)
    assert run.nodes["write"].error == "rejected by reviewer" and env.agent.writes == []


@pytest.mark.anyio
async def test_fan_out_items_are_approved_one_by_one(env: Env) -> None:
    controller = _controller(env)
    run = await controller.execute(WorkflowSpec.model_validate(FAN_OUT_WRITE), {"texts": ["a", "b"]}, REQUESTER)

    assert [item.state for item in run.nodes["write"].items] == [NodeState.WAITING_APPROVAL] * 2
    keys = [run.approvals[f"write[{i}]"].idempotency_key for i in (0, 1)]
    assert keys == [f"{run.id}:write:0:a", f"{run.id}:write:1:b"]

    first, second = run.approvals["write[0]"].payload_hash, run.approvals["write[1]"].payload_hash
    await controller.decide(run.id, "write", item=0, decision="approve", principal=REVIEWER, payload_hash=first)
    run = await controller.wait(run.id, timeout=5)
    assert run.status is RunStatus.WAITING and run.nodes["write"].state is NodeState.WAITING_APPROVAL
    await controller.decide(run.id, "write", item=1, decision="reject", principal=REVIEWER, payload_hash=second)
    run = await controller.wait(run.id, timeout=5)

    assert run.status is RunStatus.COMPLETED and run.outputs == {"written": [{"text": "wrote a"}]}
    assert [item.state for item in run.nodes["write"].items] == [NodeState.SUCCEEDED, NodeState.REJECTED]
    assert env.agent.writes == [{"text": "a"}]


# ---------------------------------------------------------------- HTTP API

ENTITLEMENTS = """
admin: ["*"]
requester: ["text.*", "record.*"]
approver: ["text.*", "record.*", "approve:record.*"]
viewer: ["text.*", "record.*"]
"""


def _headers(role: str) -> dict[str, str]:
    return {"X-Demo-Role": role}


def _api(catalog: Path, **overrides: Any) -> Iterator[tuple[TestClient, Agent]]:
    (catalog / "entitlements.yaml").write_text(ENTITLEMENTS, encoding="utf-8")
    agent = Agent()
    settings = Settings(catalog_dir=catalog, allowed_hosts=("agent.test",), planner_enabled=False, **overrides)
    with TestClient(create_app(settings, agent.http)) as client:
        admin = _headers("admin")
        assert client.post("/api/v1/capabilities", json=_contract("record.write", "write"), headers=admin).is_success
        card = build_card(BASE, _card("writer", "record.write@1", "write"))
        submitted = client.post("/api/v1/implementations/cards", json=card, headers=admin)
        assert submitted.status_code == 201, submitted.text
        approved = client.post(f"/api/v1/implementations/{submitted.json()['key']}/approve", json={}, headers=admin)
        assert approved.status_code == 200
        agent.writes.clear()
        yield client, agent


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, Agent]]:
    yield from _api(tmp_path)


def _start(client: TestClient, role: str = "requester") -> dict[str, Any]:
    body = {"workflow": copy.deepcopy(SINGLE_WRITE), "inputs": {"text": "x"}, "wait_s": 5}
    response = client.post("/api/v1/runs", json=body, headers=_headers(role))
    assert response.status_code == 201, response.text
    return response.json()


def test_direct_invoke_of_a_write_is_403(api: tuple[TestClient, Agent]) -> None:
    client, agent = api
    body = {"capability": "record.write", "input": {"text": "x"}}
    response = client.post("/api/v1/invoke", json=body, headers=_headers("approver"))
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "write capability requires a human approval bound to this payload"
    assert agent.writes == []


def test_approval_endpoints_enforce_roles_four_eyes_and_hash(api: tuple[TestClient, Agent]) -> None:
    client, agent = api
    run = _start(client)
    assert run["status"] == "waiting"
    [pending] = run["approvals"]
    path = f"/api/v1/runs/{run['id']}/approvals/write"
    approve = {"decision": "approve", "payload_hash": pending["payload_hash"]}

    assert client.get("/api/v1/approvals", headers=_headers("requester")).json() == []
    assert client.get("/api/v1/approvals", headers=_headers("viewer")).json() == []
    [listed] = client.get("/api/v1/approvals", headers=_headers("approver")).json()
    assert (listed["run_id"], listed["payload"], listed["requested_by"]) == (run["id"], {"text": "x"}, "demo-requester")

    own = client.post(path, json=approve, headers=_headers("requester"))
    assert (own.status_code, own.json()["detail"]) == (403, "four-eyes: approver must differ from requester")
    viewer = client.post(path, json=approve, headers=_headers("viewer"))
    assert (viewer.status_code, viewer.json()["detail"]) == (403, "role 'viewer' may not approve record.write")
    stale = client.post(path, json={**approve, "payload_hash": "0" * 64}, headers=_headers("approver"))
    assert stale.status_code == 409
    assert client.post(path, json={**approve, "item": 3}, headers=_headers("approver")).status_code == 404
    assert agent.writes == []

    decided = client.post(path, json=approve, headers=_headers("approver"))
    assert decided.status_code == 200, decided.text
    assert decided.json()["inputs"] is None  # the approver is not the run owner
    final = client.get(f"/api/v1/runs/{run['id']}", params={"wait_s": 5}, headers=_headers("requester")).json()
    assert (final["status"], final["outputs"], final["approvals"]) == ("completed", {"text": "wrote x"}, [])
    assert agent.writes == [{"text": "x"}]
    assert client.get("/api/v1/approvals", headers=_headers("approver")).json() == []
    audit = client.get("/api/v1/audit", headers=_headers("requester")).json()
    effect = next(step for step in audit[0]["steps"] if step["name"] == "effect")
    assert effect["detail"].startswith("write approved by demo-approver")


def test_reject_over_http_completes_rejected(api: tuple[TestClient, Agent]) -> None:
    client, agent = api
    run = _start(client)
    body = {"decision": "reject", "payload_hash": run["approvals"][0]["payload_hash"]}
    assert client.post(f"/api/v1/runs/{run['id']}/approvals/write", json=body, headers=_headers("admin")).is_success
    final = client.get(f"/api/v1/runs/{run['id']}", params={"wait_s": 5}, headers=_headers("requester")).json()
    assert (final["status"], final["nodes"]["write"]["state"]) == ("completed_rejected", "rejected")
    assert agent.writes == []


def test_self_approval_only_when_enabled(tmp_path: Path) -> None:
    for allowed in (False, True):
        catalog = tmp_path / str(allowed)
        catalog.mkdir()
        for client, agent in _api(catalog, allow_self_approval=allowed):
            run = _start(client, "admin")
            assert len(client.get("/api/v1/approvals", headers=_headers("admin")).json()) == int(allowed)
            body = {"decision": "approve", "payload_hash": run["approvals"][0]["payload_hash"]}
            response = client.post(f"/api/v1/runs/{run['id']}/approvals/write", json=body, headers=_headers("admin"))
            assert response.status_code == (200 if allowed else 403)
            assert len(agent.writes) == int(allowed)
