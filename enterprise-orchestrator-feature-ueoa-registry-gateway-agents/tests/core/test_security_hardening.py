"""Security hardening: Entra token routing, browser guards, request limits, bounded audit, run paths."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ueoa.api.app import Settings, build_services, create_app
from ueoa.contracts import WorkflowSpec
from ueoa.gateway.auth import StaticTokenProvider
from ueoa.gateway.invokers import InvocationError, Invoker
from ueoa.gateway.service import Entitlements, Gateway, GatewayDeniedError, Principal
from ueoa.orchestrator.planner import PlanResult
from ueoa.orchestrator.runner import RunController
from ueoa.registry.models import DEFAULT_ENTRA_SCOPE, UEOA_EXTENSION_URI, EntraAuth
from ueoa.registry.service import Registry, RegistryError
from ueoa_sdk.host import InvocationContext, create_agent_app

ADMIN = {"X-Demo-Role": "admin"}
USER = {"X-Demo-Role": "user"}
TEXT_SCHEMA: dict[str, Any] = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
ECHO_CONTRACT: dict[str, Any] = {
    "id": "text.echo",
    "version": "1.0.0",
    "description": "Return the input text unchanged.",
    "effects": "read",
    "input_schema": TEXT_SCHEMA,
    "output_schema": TEXT_SCHEMA,
    "samples": [{"text": "hello"}],
}
WRITE_CONTRACT: dict[str, Any] = {**ECHO_CONTRACT, "id": "record.write", "effects": "write"}
PARAMS: dict[str, Any] = {
    "id": "echo-agent",
    "owner": "platform-tests",
    "kind": "agent",
    "implementation": "deterministic",
    "binding": "a2a",
    "implements": ["text.echo@1"],
    "effects": "read",
    "timeout_s": 5,
}
SKILLS = [{"id": "echo", "name": "Echo", "description": "Echo text"}]
WORKFLOW: dict[str, Any] = {
    "id": "echo-flow",
    "version": "1.0.0",
    "goal": "Echo the text",
    "inputs": {"text": {"type": "string"}},
    "nodes": [{"id": "echo", "capability": "text.echo", "inputs": {"text": "$inputs.text"}}],
    "outputs": {"text": "$nodes.echo.output.text"},
}
HUMAN_WORKFLOW: dict[str, Any] = {
    "id": "ask",
    "version": "1.0.0",
    "goal": "Ask for text",
    "nodes": [{"id": "ask", "type": "human_input", "input_schema": TEXT_SCHEMA}],
}


def _card(interface: str, *, auth: dict[str, Any] | None = None, health_url: str | None = None) -> dict[str, Any]:
    params = dict(PARAMS)
    if auth is not None:
        params["auth"] = auth
    if health_url is not None:
        params["health_url"] = health_url
    return {
        "name": "Echo Agent",
        "description": "Returns its input.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": interface, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
        "capabilities": {"extensions": [{"uri": UEOA_EXTENSION_URI, "params": params}]},
        "skills": SKILLS,
    }


def _overlay(scope: str = DEFAULT_ENTRA_SCOPE) -> dict[str, Any]:
    return {**PARAMS, "auth": {"scheme": "entra", "scope": scope}, "a2a": {"card_path": "card"}}


class CardHost:
    """Serves a card without the UEOA extension at <url>/card; records what it was sent."""

    def __init__(self, interface: str | None = None) -> None:
        self.interface = interface
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/card"):
            interface = self.interface or str(request.url).removesuffix("/card")
            card = _card(interface)
            del card["capabilities"]
            return httpx.Response(200, json=card)
        return httpx.Response(404)


@asynccontextmanager
async def _registry(hosts: tuple[str, ...], host: CardHost | None = None) -> AsyncIterator[tuple[Registry, Invoker]]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(host or CardHost())) as http:
        invoker = Invoker(http, poll_interval_s=0.01, token_provider=StaticTokenProvider("fake-entra"))
        registry = Registry({}, invoker, http, hosts)
        registry.register_capability(ECHO_CONTRACT)
        yield registry, invoker


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------- 1. Entra tokens


@pytest.mark.anyio
async def test_entra_scope_must_be_allow_listed() -> None:
    host = CardHost()
    async with _registry(("agent.test",), host) as (registry, invoker):
        bad = {"scheme": "entra", "scope": "api://elsewhere/.default"}
        with pytest.raises(RegistryError, match=r"scope 'api://elsewhere/\.default' is not allowed"):
            await registry.register_card(_card("http://agent.test/svc", auth=bad))
        with pytest.raises(RegistryError, match="is not allowed"):
            await registry.register_from_url("http://agent.test/svc", _overlay("api://elsewhere/.default"))
        assert host.requests == []  # no token requested, no card fetched
        with pytest.raises(InvocationError) as denied:
            await invoker.auth_headers(EntraAuth(scheme="entra", scope="api://x/.default"), "k", "http://agent.test")
        assert denied.value.code == "not_allowed"
        impl = await registry.register_card(_card("http://agent.test/svc", auth={"scheme": "entra"}))
        assert impl.key == "echo-agent@1.0.0"


@pytest.mark.anyio
async def test_entra_hosts_need_an_exact_allow_list_entry_or_loopback() -> None:
    async with _registry(("*.test", "local*")) as (registry, _):
        entra = {"scheme": "entra"}
        with pytest.raises(RegistryError, match="exact"):
            await registry.register_card(_card("http://agent.test/svc", auth=entra))
        with pytest.raises(RegistryError, match="exact"):
            await registry.register_from_url("http://agent.test/svc", _overlay())
        assert (await registry.register_card(_card("http://localhost:9/svc", auth=entra))).source == "submitted"
        plain = await registry.register_card({**_card("http://agent.test/svc"), "version": "2.0.0"})
        assert plain.key == "echo-agent@2.0.0"  # wildcards still admit agents that get no Entra token


@pytest.mark.anyio
async def test_entra_endpoints_are_pinned_to_the_registration_host() -> None:
    host = CardHost(interface="http://other.test/svc")
    async with _registry(("agent.test", "other.test"), host) as (registry, _):
        with pytest.raises(RegistryError, match=r"must stay on host 'agent\.test'"):
            await registry.register_from_url("http://agent.test/svc", _overlay())
        card = _card("http://agent.test/svc", auth={"scheme": "entra"}, health_url="http://other.test/health")
        with pytest.raises(RegistryError, match=r"must stay on host 'agent\.test'"):
            await registry.register_card(card)


@pytest.mark.anyio
async def test_static_token_is_sent_to_loopback_hosts_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "azure.identity.aio", None)
    host = CardHost()
    settings = Settings(
        catalog_dir=None,
        allowed_hosts=("localhost", "agent.example"),
        entra_static_token="dev-tok",  # noqa: S106 - fake
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(host)) as http:
        with caplog.at_level(logging.WARNING):
            services = await build_services(settings, http)
        assert "UEOA_ENTRA_STATIC_TOKEN" in caplog.text
        services.registry.register_capability(ECHO_CONTRACT)
        await services.registry.register_from_url("http://localhost:9/svc", _overlay())
        assert host.requests[0].headers["Authorization"] == "Bearer dev-tok"
        sent = len(host.requests)
        with pytest.raises(RegistryError, match="could not authenticate"):
            await services.registry.register_from_url("http://agent.example/svc", _overlay())
    assert len(host.requests) == sent
    assert all(r.url.host == "localhost" for r in host.requests)


# ---------------------------------------------------------------- API fixtures


class ScriptedLLM:
    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        workflow = json.loads(json.dumps(WORKFLOW))
        workflow["nodes"][0]["timeout_s"] = 1  # below the card timeout: the run must raise it
        return {"rationale": "One echo step.", "workflow": workflow}


async def _echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


@contextmanager
def _api(**overrides: Any) -> Iterator[TestClient]:
    agent = create_agent_app(_card("http://echo.test"), {"text.echo": _echo}, "http://echo.test")
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=agent))
    settings = Settings(catalog_dir=None, allowed_hosts=("echo.test",), **overrides)
    with TestClient(create_app(settings, http, planner_llm=ScriptedLLM())) as client:
        assert client.post("/api/v1/capabilities", json=ECHO_CONTRACT, headers=ADMIN).status_code == 201
        registered = client.post("/api/v1/implementations", json={"url": "http://echo.test"}, headers=ADMIN)
        assert registered.status_code == 201, registered.text
        key = registered.json()["key"]
        assert client.post(f"/api/v1/implementations/{key}/approve", json={}, headers=ADMIN).status_code == 200
        yield client


def _mcp(client: TestClient, method: str, params: dict[str, Any], **headers: str) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return client.post("/mcp", json=body, headers={**USER, **headers})


# ---------------------------------------------------------------- 2. browser guards


def test_untrusted_host_header_is_rejected() -> None:
    with _api() as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/health", headers={"Host": "evil.example"}).status_code == 400
        assert client.get("/api/v1/health", headers={"Host": "localhost:8000"}).status_code == 200


def test_writes_need_a_local_origin_on_the_server_port() -> None:
    body = {"workflow": WORKFLOW, "inputs": {"text": "x"}}
    with _api() as client:
        for origin in ("https://evil.example", "http://localhost:3000", "null"):
            response = client.post("/api/v1/runs", json=body, headers={**USER, "Origin": origin})
            assert response.status_code == 403, origin
        assert client.get("/api/v1/runs", headers={**USER, "Origin": "https://evil.example"}).status_code == 200
        for origin in ("http://localhost", "http://127.0.0.1:80"):
            assert client.post("/api/v1/runs", json=body, headers={**USER, "Origin": origin}).status_code == 201
        assert client.post("/api/v1/runs", json=body, headers=USER).status_code == 201  # non-browser client
        assert _mcp(client, "ping", {}, Origin="http://localhost:3000").status_code == 403


def test_configured_origins_must_match_exactly() -> None:
    body = {"workflow": WORKFLOW, "inputs": {"text": "x"}}
    with _api(allowed_origins=("http://localhost:3000",)) as client:
        assert client.post("/api/v1/runs", json=body, headers={**USER, "Origin": "http://localhost:3000"}).is_success
        for origin in ("http://localhost", "https://localhost:3000", "http://localhost:3001"):
            response = client.post("/api/v1/runs", json=body, headers={**USER, "Origin": origin})
            assert response.status_code == 403, origin
        assert _mcp(client, "ping", {}, Origin="http://localhost:3000").status_code == 200


def test_mcp_post_requires_json_content_type() -> None:
    with _api() as client:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        for content_type in ("text/plain", "application/x-www-form-urlencoded"):
            response = client.post("/mcp", content=body, headers={**USER, "Content-Type": content_type})
            assert response.status_code == 415
        ok = client.post("/mcp", content=body, headers={**USER, "Content-Type": "application/json; charset=utf-8"})
        assert ok.status_code == 200


# ---------------------------------------------------------------- 3. request size and workflow limits


def test_oversized_bodies_are_413() -> None:
    with _api(max_body_bytes=2000) as client:
        big = {**ECHO_CONTRACT, "id": "text.big", "description": "x" * 3000}
        assert client.post("/api/v1/capabilities", json=big, headers=ADMIN).status_code == 413
        streamed = json.dumps(big).encode()
        chunks = iter([streamed[:1500], streamed[1500:]])
        headers = {**ADMIN, "Content-Type": "application/json"}
        response = client.post("/api/v1/capabilities", content=chunks, headers=headers)
        assert response.status_code == 413
        rpc = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"pad": "x" * 3000}}
        assert client.post("/mcp", json=rpc, headers=USER).status_code == 413
        assert "text.big" not in {c["id"] for c in client.get("/api/v1/capabilities").json()}


def test_workflow_node_count_and_timeout_are_capped() -> None:
    node = WORKFLOW["nodes"][0]
    many = [{**node, "id": f"n{i}"} for i in range(26)]
    with pytest.raises(ValidationError, match="at most 25"):
        WorkflowSpec.model_validate({**WORKFLOW, "nodes": many})
    WorkflowSpec.model_validate({**WORKFLOW, "nodes": many[:25]})
    with pytest.raises(ValidationError, match="less than or equal to 900"):
        WorkflowSpec.model_validate({**WORKFLOW, "nodes": [{**node, "timeout_s": 901}]})
    WorkflowSpec.model_validate({**WORKFLOW, "nodes": [{**node, "timeout_s": 900}]})


# ---------------------------------------------------------------- 4. bounded audit


@pytest.mark.anyio
async def test_audit_keeps_only_the_latest_records() -> None:
    async with httpx.AsyncClient() as http:
        invoker = Invoker(http)
        gateway = Gateway(Registry({}, invoker, http, ()), invoker, Entitlements({"user": ["*"]}), max_audit=3)
        for index in range(5):
            with pytest.raises(GatewayDeniedError):
                await gateway.invoke(Principal("p", "user"), f"missing.cap{index}", {})
    assert [record.capability for record in reversed(gateway.audit)] == ["missing.cap4", "missing.cap3", "missing.cap2"]


def test_api_audit_is_bounded_and_listed_newest_first() -> None:
    with _api() as client:
        services = client.app.state.services  # type: ignore[attr-defined]
        assert services.gateway.audit.maxlen == 2000
        for text in ("a", "b"):
            client.post("/api/v1/invoke", json={"capability": "text.echo", "input": {"text": text}}, headers=USER)
        assert [r["output"]["text"] for r in client.get("/api/v1/audit", headers=USER).json()] == ["b", "a"]


# ---------------------------------------------------------------- 5. MCP run path


def test_mcp_run_goal_gets_card_timeout_headroom() -> None:
    with _api() as client:
        arguments = {"goal": "echo", "inputs": {"text": "hi"}, "wait_s": 10}
        response = _mcp(client, "tools/call", {"name": "ueoa_run_goal", "arguments": arguments})
        result = response.json()["result"]
        assert result["isError"] is False, result
        run = json.loads(result["content"][0]["text"])
    assert (run["status"], run["outputs"]) == ("completed", {"text": "hi"})
    assert run["nodes"]["echo"]["timeout_s"] == 20  # card timeout 5s + 15s headroom


def test_rest_run_timeouts_get_the_same_headroom() -> None:
    low = json.loads(json.dumps(WORKFLOW))
    low["nodes"][0]["timeout_s"] = 1
    with _api() as client:
        for workflow in (WORKFLOW, low):
            body = {"workflow": workflow, "inputs": {"text": "x"}, "wait_s": 10}
            run = client.post("/api/v1/runs", json=body, headers=USER).json()
            assert (run["status"], run["nodes"]["echo"]["timeout_s"]) == ("completed", 20)


class PlannerIgnoringAvailability:
    """A planner that returns a capability the caller may not use (the run path must still refuse it)."""

    async def plan(self, goal: str, inputs: dict[str, Any], capabilities: Any, available: list[str]) -> PlanResult:
        spec = WorkflowSpec.model_validate(
            {**WORKFLOW, "nodes": [{**WORKFLOW["nodes"][0], "capability": "record.write"}]}
        )
        return PlanResult(spec=spec, rationale="r", attempts=1, issues=[], considered=[])


def test_mcp_run_goal_refuses_unavailable_capabilities() -> None:
    with _api() as client:
        assert client.post("/api/v1/capabilities", json=WRITE_CONTRACT, headers=ADMIN).status_code == 201
        services = client.app.state.services  # type: ignore[attr-defined]
        services.planner = PlannerIgnoringAvailability()
        arguments = {"goal": "write", "inputs": {"text": "x"}}
        result = _mcp(client, "tools/call", {"name": "ueoa_run_goal", "arguments": arguments}).json()["result"]
        assert result["isError"] is True
        assert "record.write" in json.loads(result["content"][0]["text"])["error"]
        assert client.get("/api/v1/runs", headers=USER).json() == []
        assert list(services.gateway.audit) == []


# ---------------------------------------------------------------- 6. cancellation and shutdown


def test_cancel_is_owner_or_admin_only() -> None:
    with _api() as client:
        run = client.post("/api/v1/runs", json={"workflow": HUMAN_WORKFLOW}, headers=ADMIN).json()
        url = f"/api/v1/runs/{run['id']}/cancel"
        assert client.post(url, headers=USER).status_code == 404
        assert client.post("/api/v1/runs/no-such-run/cancel", headers=ADMIN).status_code == 404
        response = client.post(url, headers=ADMIN)
        if hasattr(RunController, "cancel"):
            assert response.status_code == 200, response.text
            assert response.json()["id"] == run["id"]
        else:
            assert response.status_code == 501


def test_shutdown_closes_the_run_controller() -> None:
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    with _api() as client:
        client.app.state.services.runs.close = close  # type: ignore[attr-defined]
        assert closed == []
    assert closed == [True]
