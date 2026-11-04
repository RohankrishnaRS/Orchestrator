"""Routing view on plans and runs, and the agent cards API, with a scripted planner and SDK-hosted A2A agents."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from ueoa.api.app import Settings, create_app
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa_sdk.host import InvocationContext, create_agent_app

ADMIN = {"X-Demo-Role": "admin"}
USER = {"X-Demo-Role": "user"}
TEXT_SCHEMA: dict[str, Any] = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
AGENTS = {
    "echo": ("text.echo", "Echo Agent"),
    "upper": ("text.upper", "Upper Agent"),
    "restricted": ("text.restricted", "Restricted Agent"),
}
WORKFLOW: dict[str, Any] = {
    "id": "echo-each",
    "version": "1.0.0",
    "goal": "Echo the text and every item",
    "inputs": {"text": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}}},
    "nodes": [
        {"id": "first", "capability": "text.echo", "inputs": {"text": "$inputs.text"}},
        {"id": "each", "capability": "text.echo", "for_each": "$inputs.items", "inputs": {"text": "$item"}},
    ],
    "outputs": {"text": "$nodes.first.output.text"},
}
INPUTS = {"text": "hi", "items": ["a", "b"]}


def _contract(capability_id: str) -> dict[str, Any]:
    return {
        "id": capability_id,
        "version": "1.0.0",
        "description": f"Text operation {capability_id}.",
        "effects": "read",
        "input_schema": {**TEXT_SCHEMA, "additionalProperties": False},
        "output_schema": TEXT_SCHEMA,
        "samples": [{"text": "hello"}],
    }


def _card(agent_id: str, capability_id: str, name: str) -> dict[str, Any]:
    params = {
        "id": f"{agent_id}-agent",
        "owner": "platform-tests",
        "kind": "agent",
        "implementation": "deterministic",
        "binding": "a2a",
        "implements": [f"{capability_id}@1"],
        "effects": "read",
        "timeout_s": 5,
    }
    return {
        "name": name,
        "description": f"{name} for tests.",
        "version": "1.0.0",
        "capabilities": {"extensions": [{"uri": UEOA_EXTENSION_URI, "params": params}]},
        "skills": [{"id": agent_id, "name": name, "description": f"Runs {capability_id}"}],
    }


class ScriptedLLM:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        return copy.deepcopy(self.reply)


async def _echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


def _by_host(apps: dict[str, ASGIApp]) -> ASGIApp:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        host = dict(scope["headers"]).get(b"host", b"").decode().split(":")[0]
        await apps[host](scope, receive, send)

    return app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    (tmp_path / "entitlements.yaml").write_text(
        yaml.safe_dump({"admin": ["*"], "user": ["text.echo", "text.upper"]}), encoding="utf-8"
    )
    apps = {
        f"{agent}.test": create_agent_app(_card(agent, capability, name), {capability: _echo}, f"http://{agent}.test")
        for agent, (capability, name) in AGENTS.items()
    }
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_by_host(apps)))
    settings = Settings(catalog_dir=tmp_path, allowed_hosts=tuple(apps), planner_enabled=True)
    llm = ScriptedLLM({"rationale": "Echo once and per item.", "workflow": WORKFLOW})
    with TestClient(create_app(settings, http, planner_llm=llm)) as test_client:
        for agent, (capability, _name) in AGENTS.items():
            assert (
                test_client.post("/api/v1/capabilities", json=_contract(capability), headers=ADMIN).status_code == 201
            )
            registered = test_client.post(
                "/api/v1/implementations", json={"url": f"http://{agent}.test"}, headers=ADMIN
            )
            assert registered.status_code == 201, registered.text
            key = registered.json()["key"]
            assert test_client.post(f"/api/v1/implementations/{key}/approve", json={}, headers=ADMIN).status_code == 200
        yield test_client


SELECTED = [
    {
        "node": "first",
        "capability": "text.echo",
        "implementation": "echo-agent@1.0.0",
        "agent_name": "Echo Agent",
        "binding": "a2a",
        "fan_out": False,
    },
    {
        "node": "each",
        "capability": "text.echo",
        "implementation": "echo-agent@1.0.0",
        "agent_name": "Echo Agent",
        "binding": "a2a",
        "fan_out": True,
    },
]


def _left_out(capability: str, agent: str, reason: str) -> dict[str, Any]:
    _, name = AGENTS[agent]
    return {
        "capability": capability,
        "implementation": f"{agent}-agent@1.0.0",
        "agent_name": name,
        "binding": "a2a",
        "reason": reason,
    }


def test_plan_routing_lists_selected_and_excluded_agents(client: TestClient) -> None:
    response = client.post("/api/v1/plans", json={"goal": "echo things", "inputs": INPUTS}, headers=USER)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["ok"], body["available"]) == (True, ["text.echo", "text.upper"])
    routing = body["routing"]
    assert routing["selected"] == SELECTED
    assert routing["not_selected"] == [
        _left_out("text.upper", "upper", "not relevant to the goal (planner)"),
        _left_out("text.restricted", "restricted", "not permitted for role"),
    ]
    assert routing["considered"] == body["considered"]
    assert set(routing["considered"]) <= set(body["available"])


def test_admin_sees_every_unused_agent_as_not_relevant(client: TestClient) -> None:
    body = client.post("/api/v1/plans", json={"goal": "echo things", "inputs": INPUTS}, headers=ADMIN).json()
    assert {(n["capability"], n["reason"]) for n in body["routing"]["not_selected"]} == {
        ("text.restricted", "not relevant to the goal (planner)"),
        ("text.upper", "not relevant to the goal (planner)"),
    }


def test_goal_run_carries_routing_in_plan_metadata(client: TestClient) -> None:
    body = {"goal": "echo things", "inputs": INPUTS, "wait_s": 10}
    response = client.post("/api/v1/runs", json=body, headers=USER)
    assert response.status_code == 201, response.text
    run = response.json()
    assert (run["status"], run["outputs"]) == ("completed", {"text": "hi"})
    assert run["plan"]["routing"]["selected"] == SELECTED
    reasons = {n["capability"]: n["reason"] for n in run["plan"]["routing"]["not_selected"]}
    assert reasons == {"text.upper": "not relevant to the goal (planner)", "text.restricted": "not permitted for role"}
    again = client.get(f"/api/v1/runs/{run['id']}", headers=USER).json()
    assert again["plan"]["routing"] == run["plan"]["routing"]


def test_workflow_run_routing_marks_unused_agents(client: TestClient) -> None:
    run = client.post("/api/v1/runs", json={"workflow": WORKFLOW, "inputs": INPUTS}, headers=USER).json()
    routing = run["plan"]["routing"]
    assert routing["selected"] == SELECTED
    assert routing["considered"] == []
    assert _left_out("text.upper", "upper", "not used by the workflow") in routing["not_selected"]


def test_cards_api_lists_every_registered_card(client: TestClient) -> None:
    cards = client.get("/api/v1/cards").json()
    assert [c["key"] for c in cards] == ["echo-agent@1.0.0", "restricted-agent@1.0.0", "upper-agent@1.0.0"]
    echo = cards[0]
    assert {k: echo[k] for k in ("name", "url", "binding", "implements", "status")} == {
        "name": "Echo Agent",
        "url": "http://echo.test/.well-known/agent-card.json",
        "binding": "a2a",
        "implements": ["text.echo@1"],
        "status": "approved",
    }
    assert isinstance(echo["priority"], int)
    assert echo["description"] == "Echo Agent for tests."
    assert echo["skills"] == [{"id": "echo", "name": "Echo Agent", "description": "Runs text.echo"}]
    assert echo["card"]["name"] == "Echo Agent"
    assert echo["card"]["skills"][0]["id"] == "echo"
