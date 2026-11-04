"""MCP gateway over HTTP: handshake, tool listing from the gateway's view, calls, denials, notifications, Origin."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Header

from ueoa.api.mcp import create_mcp_router
from ueoa.gateway.invokers import Invoker
from ueoa.gateway.service import Entitlements, Gateway, Principal
from ueoa.orchestrator.planner import Planner
from ueoa.orchestrator.runner import RunController
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa.registry.service import Registry
from ueoa_sdk.host import InvocationContext, create_agent_app

pytestmark = pytest.mark.anyio

AGENT = "http://echo.test"
ECHO_CONTRACT: dict[str, Any] = {
    "id": "text.echo",
    "version": "1.0.0",
    "description": "Return the input text unchanged.",
    "effects": "read",
    "selection_doc": {"use_when": "the text must come back as is."},
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
    },
    "output_schema": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
    "samples": [{"text": "hello"}],
}
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
PLAN_REPLY: dict[str, Any] = {
    "rationale": "echo the text",
    "workflow": {
        "id": "echo-plan",
        "version": "1.0.0",
        "goal": "echo it",
        "inputs": {"text": {"type": "string"}},
        "nodes": [{"id": "echo", "capability": "text.echo", "inputs": {"text": "$inputs.text"}}],
        "outputs": {"text": "$nodes.echo.output.text"},
    },
}


class ScriptedLLM:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        return json.loads(json.dumps(self.reply))


@dataclass
class Services:
    registry: Registry
    gateway: Gateway
    entitlements: Entitlements
    runs: RunController | None = None
    planner: Planner | None = None


async def _echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def services() -> AsyncIterator[Services]:
    agent = create_agent_app(ECHO_CARD, {"text.echo": _echo}, AGENT)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=agent)) as http:
        invoker = Invoker(http)
        registry = Registry({}, invoker, http, ("echo.test",))
        registry.register_capability(ECHO_CONTRACT)
        registry.register_capability({**ECHO_CONTRACT, "id": "text.unapproved", "selection_doc": None})
        impl = await registry.register_from_url(AGENT)
        registry.approve(impl.key)
        entitlements = Entitlements({"user": ["text.*"], "guest": []})
        yield Services(registry, Gateway(registry, invoker, entitlements), entitlements)


@pytest.fixture
async def client(services: Services) -> AsyncIterator[httpx.AsyncClient]:
    def principal(x_demo_role: str = Header("user")) -> Principal:
        return Principal(f"demo-{x_demo_role}", x_demo_role)

    app = FastAPI()
    app.include_router(create_mcp_router(lambda: services, principal))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ueoa.test") as http:
        yield http


async def _rpc(
    client: httpx.AsyncClient, method: str, params: dict[str, Any] | None = None, role: str = "user"
) -> dict[str, Any]:
    response = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 7, "method": method, "params": params or {}},
        headers={"X-Demo-Role": role},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["jsonrpc"] == "2.0" and body["id"] == 7
    return body


async def test_initialize_negotiates_version(client: httpx.AsyncClient) -> None:
    result = (await _rpc(client, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}}))["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "ueoa-gateway"
    fallback = (await _rpc(client, "initialize", {"protocolVersion": "1999-01-01"}))["result"]
    assert fallback["protocolVersion"] == "2025-06-18"
    assert (await _rpc(client, "ping"))["result"] == {}


async def test_notifications_get_202_and_other_methods_are_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert (response.status_code, response.content) == (202, b"")
    assert (await _rpc(client, "resources/list"))["error"]["code"] == -32601
    batch = await client.post("/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}])
    assert batch.status_code == 400 and batch.json()["error"]["code"] == -32600
    assert (await client.get("/mcp")).status_code == 405


async def test_tools_list_reflects_the_gateway_view(client: httpx.AsyncClient) -> None:
    tools = (await _rpc(client, "tools/list"))["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["text__echo"]
    tool = tools[0]
    assert tool["description"] == "Return the input text unchanged. Use when: the text must come back as is."
    assert tool["inputSchema"] == ECHO_CONTRACT["input_schema"]
    assert tool["outputSchema"] == ECHO_CONTRACT["output_schema"]
    assert tool["annotations"] == {"readOnlyHint": True}
    assert (await _rpc(client, "tools/list", role="guest"))["result"]["tools"] == []


async def test_tools_call_success_failure_and_unknown(client: httpx.AsyncClient) -> None:
    ok = (await _rpc(client, "tools/call", {"name": "text__echo", "arguments": {"text": "hi"}}))["result"]
    assert ok == {
        "content": [{"type": "text", "text": '{"text": "hi"}'}],
        "structuredContent": {"text": "hi"},
        "isError": False,
    }

    bad = (await _rpc(client, "tools/call", {"name": "text__echo", "arguments": {"nope": 1}}))["result"]
    assert bad["isError"] is True and "input_schema: deny" in bad["content"][0]["text"]

    for name in ("text__unapproved", "missing__tool"):
        error = (await _rpc(client, "tools/call", {"name": name, "arguments": {}}))["error"]
        assert error["code"] == -32602
    denied = await _rpc(client, "tools/call", {"name": "text__echo", "arguments": {"text": "x"}}, role="guest")
    assert denied["error"]["code"] == -32602


async def test_origin_must_be_local_on_the_server_port(client: httpx.AsyncClient) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert (await client.post("/mcp", json=body, headers={"Origin": "https://evil.example"})).status_code == 403
    assert (await client.post("/mcp", json=body, headers={"Origin": "http://localhost:6274"})).status_code == 403
    assert (await client.post("/mcp", json=body, headers={"Origin": "http://127.0.0.1"})).status_code == 200
    assert (await client.post("/mcp", json=body, headers={"Origin": "http://localhost:80"})).status_code == 200


async def test_meta_tools_plan_and_run_a_goal(services: Services, client: httpx.AsyncClient) -> None:
    services.planner = Planner(ScriptedLLM(PLAN_REPLY))
    services.runs = RunController(services.gateway, lambda: services.registry.capabilities, max_backoff_s=0)
    names = [tool["name"] for tool in (await _rpc(client, "tools/list"))["result"]["tools"]]
    assert names == ["text__echo", "ueoa_plan", "ueoa_run_goal"]

    args = {"goal": "echo it", "inputs": {"text": "hi"}}
    plan = (await _rpc(client, "tools/call", {"name": "ueoa_plan", "arguments": args}))["result"]
    assert plan["isError"] is False
    assert json.loads(plan["content"][0]["text"])["workflow"]["nodes"][0]["capability"] == "text.echo"

    run = (await _rpc(client, "tools/call", {"name": "ueoa_run_goal", "arguments": {**args, "wait_s": 5}}))["result"]
    view = json.loads(run["content"][0]["text"])
    assert run["isError"] is False
    assert (view["status"], view["outputs"]) == ("completed", {"text": "hi"})

    missing_goal = await _rpc(client, "tools/call", {"name": "ueoa_plan", "arguments": {}})
    assert missing_goal["error"]["code"] == -32602
