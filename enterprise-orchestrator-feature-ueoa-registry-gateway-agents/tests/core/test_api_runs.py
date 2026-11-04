"""Plans and runs over the HTTP API, with a scripted planner model and an SDK-hosted A2A agent."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa_sdk.host import InvocationContext, create_agent_app

ADMIN = {"X-Demo-Role": "admin"}
USER = {"X-Demo-Role": "user"}
TEXT_SCHEMA: dict[str, Any] = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
ECHO_CONTRACT: dict[str, Any] = {
    "id": "text.echo",
    "version": "1.0.0",
    "description": "Return the input text unchanged.",
    "effects": "read",
    "input_schema": {**TEXT_SCHEMA, "additionalProperties": False},
    "output_schema": TEXT_SCHEMA,
    "samples": [{"text": "hello"}],
}
WRITE_CONTRACT: dict[str, Any] = {**ECHO_CONTRACT, "id": "record.write", "effects": "write"}
ECHO_CARD: dict[str, Any] = {
    "name": "Echo Agent",
    "description": "Returns its input.",
    "version": "1.0.0",
    "capabilities": {
        "extensions": [
            {
                "uri": UEOA_EXTENSION_URI,
                "params": {
                    "id": "echo-agent",
                    "owner": "platform-tests",
                    "kind": "agent",
                    "implementation": "deterministic",
                    "binding": "a2a",
                    "implements": ["text.echo@1"],
                    "effects": "read",
                    "timeout_s": 5,
                },
            }
        ]
    },
    "skills": [{"id": "echo", "name": "Echo", "description": "Echo text"}],
}
WORKFLOW: dict[str, Any] = {
    "id": "echo-flow",
    "version": "1.0.0",
    "goal": "Echo the text",
    "inputs": {"text": {"type": "string"}},
    "nodes": [{"id": "echo", "capability": "text.echo", "inputs": {"text": "$inputs.text"}}],
    "outputs": {"text": "$nodes.echo.output.text"},
}
HUMAN_WORKFLOW: dict[str, Any] = {
    "id": "ask-then-echo",
    "version": "1.0.0",
    "goal": "Ask for text, then echo it",
    "nodes": [
        {"id": "ask", "type": "human_input", "input_schema": TEXT_SCHEMA},
        {"id": "echo", "capability": "text.echo", "depends_on": ["ask"], "inputs": {"text": "$nodes.ask.output.text"}},
    ],
    "outputs": {"text": "$nodes.echo.output.text"},
}


class ScriptedLLM:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        return copy.deepcopy(self.reply)


async def _echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


def _client(planner_llm: ScriptedLLM | None, *, planner_enabled: bool = True) -> Iterator[TestClient]:
    agent = create_agent_app(ECHO_CARD, {"text.echo": _echo}, "http://echo.test")
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=agent))
    settings = Settings(catalog_dir=None, allowed_hosts=("echo.test",), planner_enabled=planner_enabled)
    with TestClient(create_app(settings, http, planner_llm=planner_llm)) as client:
        assert client.post("/api/v1/capabilities", json=ECHO_CONTRACT, headers=ADMIN).status_code == 201
        registered = client.post("/api/v1/implementations", json={"url": "http://echo.test"}, headers=ADMIN)
        assert registered.status_code == 201, registered.text
        key = registered.json()["key"]
        assert client.post(f"/api/v1/implementations/{key}/approve", json={}, headers=ADMIN).status_code == 200
        yield client


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from _client(ScriptedLLM({"rationale": "One echo step.", "workflow": WORKFLOW}))


def test_health_reports_planner_and_catalog(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {
        "status": "ok",
        "planner": True,
        "capabilities": 1,
        "implementations": 1,
        "approved": 1,
        "default_role": "user",
    }


def test_plan_returns_workflow_and_available(client: TestClient) -> None:
    response = client.post("/api/v1/plans", json={"goal": "echo hello", "inputs": {"text": "hello"}}, headers=USER)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["ok"], body["available"], body["rationale"]) == (True, ["text.echo"], "One echo step.")
    assert [n["capability"] for n in body["workflow"]["nodes"]] == ["text.echo"]
    assert client.post("/api/v1/plans", json={"goal": ""}, headers=USER).status_code == 422


def test_run_from_goal_completes(client: TestClient) -> None:
    body = {"goal": "echo hello", "inputs": {"text": "hello"}, "wait_s": 10}
    response = client.post("/api/v1/runs", json=body, headers=USER)
    assert response.status_code == 201, response.text
    run = response.json()
    assert (run["status"], run["outputs"]) == ("completed", {"text": "hello"})
    assert run["plan"]["planned_by"] == "planner"
    assert run["nodes"]["echo"]["implementation"] == "echo-agent@1.0.0"


def test_run_from_workflow_and_long_poll(client: TestClient) -> None:
    response = client.post("/api/v1/runs", json={"workflow": WORKFLOW, "inputs": {"text": "hi"}}, headers=USER)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["plan"]["planned_by"] == "user"
    assert run["workflow"]["nodes"][0]["timeout_s"] == 20  # card timeout 5s + 15s headroom
    final = client.get(f"/api/v1/runs/{run['id']}", params={"wait_s": 10}, headers=USER).json()
    assert (final["status"], final["outputs"]) == ("completed", {"text": "hi"})


@pytest.mark.parametrize(
    "body",
    [
        {"inputs": {}},
        {"goal": "x", "workflow": WORKFLOW},
        {"workflow": {"id": "Bad Id", "nodes": []}},
        {"workflow": WORKFLOW, "inputs": {}},
    ],
)
def test_bad_run_requests_are_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/api/v1/runs", json=body, headers=USER).status_code == 422


def test_unavailable_capability_is_403(client: TestClient) -> None:
    assert client.post("/api/v1/capabilities", json=WRITE_CONTRACT, headers=ADMIN).status_code == 201
    workflow = copy.deepcopy(WORKFLOW)
    workflow["nodes"][0]["capability"] = "record.write"
    response = client.post("/api/v1/runs", json={"workflow": workflow, "inputs": {"text": "x"}}, headers=USER)
    assert response.status_code == 403
    assert "record.write" in response.json()["detail"]


def test_runs_and_audit_are_scoped_to_their_owner(client: TestClient) -> None:
    body = {"workflow": WORKFLOW, "inputs": {"text": "a"}, "wait_s": 10}
    admin_run = client.post("/api/v1/runs", json=body, headers=ADMIN).json()
    user_run = client.post("/api/v1/runs", json=body, headers=USER).json()
    assert [r["id"] for r in client.get("/api/v1/runs", headers=USER).json()] == [user_run["id"]]
    assert [r["id"] for r in client.get("/api/v1/runs", headers=ADMIN).json()] == [user_run["id"], admin_run["id"]]
    assert client.get(f"/api/v1/runs/{admin_run['id']}", headers=USER).status_code == 404
    assert client.get(f"/api/v1/runs/{user_run['id']}", headers=ADMIN).status_code == 200
    assert client.get("/api/v1/runs/no-such-run", headers=USER).status_code == 404
    user_audit = {r["principal"]["name"] for r in client.get("/api/v1/audit", headers=USER).json()}
    admin_audit = {r["principal"]["name"] for r in client.get("/api/v1/audit", headers=ADMIN).json()}
    assert (user_audit, admin_audit) == ({"demo-user"}, {"demo-user", "demo-admin"})


def test_human_input_completes_run(client: TestClient) -> None:
    run = client.post("/api/v1/runs", json={"workflow": HUMAN_WORKFLOW, "wait_s": 5}, headers=USER).json()
    assert (run["status"], run["nodes"]["ask"]["state"]) == ("waiting", "waiting_input")
    url = f"/api/v1/runs/{run['id']}/inputs/ask"
    assert client.post(url, json={"value": {"wrong": 1}}, headers=USER).status_code == 422
    other = client.post("/api/v1/runs", json={"workflow": HUMAN_WORKFLOW}, headers=ADMIN).json()
    other_url = f"/api/v1/runs/{other['id']}/inputs/ask"
    assert client.post(other_url, json={"value": {"text": "x"}}, headers=USER).status_code == 404
    response = client.post(url, json={"value": {"text": "from a person"}}, headers=USER)
    assert response.status_code == 200, response.text
    final = client.get(f"/api/v1/runs/{run['id']}", params={"wait_s": 10}, headers=USER).json()
    assert (final["status"], final["outputs"]) == ("completed", {"text": "from a person"})
    assert client.post(url, json={"value": {"text": "again"}}, headers=USER).status_code == 422


def test_without_planner_plans_are_503_but_workflows_run() -> None:
    for client in _client(None, planner_enabled=False):
        assert client.get("/api/v1/health").json()["planner"] is False
        assert client.post("/api/v1/plans", json={"goal": "echo"}, headers=USER).status_code == 503
        assert client.post("/api/v1/runs", json={"goal": "echo"}, headers=USER).status_code == 503
        body = {"workflow": WORKFLOW, "inputs": {"text": "ok"}, "wait_s": 10}
        assert client.post("/api/v1/runs", json=body, headers=USER).json()["outputs"] == {"text": "ok"}


def test_failed_plan_is_422() -> None:
    for client in _client(ScriptedLLM({"cannot_plan": True, "rationale": "nothing fits"})):
        response = client.post("/api/v1/runs", json={"goal": "do the impossible"}, headers=USER)
        assert response.status_code == 422
        assert response.json()["detail"]["ok"] is False
