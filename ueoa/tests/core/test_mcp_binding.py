"""Outbound MCP binding: a remote MCP server tool registered by card and invoked through the gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from ueoa.gateway.auth import StaticTokenProvider
from ueoa.gateway.invokers import InvocationError, Invoker
from ueoa.gateway.service import Entitlements, Gateway, GatewayFailedError, Principal
from ueoa.registry.models import ImplementationStatus, with_extension
from ueoa.registry.service import Registry, RegistryError

pytestmark = pytest.mark.anyio

URL = "http://mcp.test/mcp"
SECRET = "s3cret"  # noqa: S105 - fake value accepted only by the in-test server
SCOPE = "api://test/.default"
USER = Principal("tester", "user")
ALLOWED = ("mcp.test", "other.test", "*.wild.test")
TEXT_SCHEMA: dict[str, Any] = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
CONTRACT: dict[str, Any] = {
    "id": "text.upper",
    "version": "1.0.0",
    "description": "Neutral upper-casing.",
    "effects": "read",
    "input_schema": TEXT_SCHEMA,
    "output_schema": TEXT_SCHEMA,
    "samples": [{"text": "hello"}],
}
TOOL: dict[str, Any] = {"name": "upper", "description": "Upper-case text", "inputSchema": TEXT_SCHEMA}


@dataclass
class McpServer:
    """Streamable HTTP MCP server with one neutral tool; answers as JSON or SSE and tracks sessions."""

    mode: str = "json"  # json | sse
    output: str = "structured"  # structured | json_text | plain
    status: int | None = None  # forced HTTP status for tools/call
    token: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=lambda: [TOOL])
    sessions: set[str] = field(default_factory=lambda: set())
    methods: list[str] = field(default_factory=lambda: [])
    requests: list[httpx.Headers] = field(default_factory=lambda: [])

    def reply(self, request_id: Any, result: Any) -> Response:
        message = {"jsonrpc": "2.0", "id": request_id, "result": result}
        if self.mode == "sse":
            progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
            stream = f"event: message\ndata: {json.dumps(progress)}\n\nevent: message\ndata: {json.dumps(message)}\n\n"
            return Response(stream, media_type="text/event-stream")
        return JSONResponse(message)

    def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = str(arguments.get("text", ""))
        if text == "fail":
            return {"content": [{"type": "text", "text": "tool exploded"}], "isError": True}
        value = {"text": text.upper()}
        if self.output == "plain":
            return {"content": [{"type": "text", "text": text.upper()}]}
        result: dict[str, Any] = {"content": [{"type": "text", "text": json.dumps(value)}], "isError": False}
        if self.output == "structured":
            result["structuredContent"] = value
        return result

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/mcp")
        async def mcp(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
            self.requests.append(request.headers)
            if self.token and request.headers.get("authorization") != f"Bearer {self.token}":
                return Response(status_code=401)
            body = await request.json()
            method, request_id, params = body["method"], body.get("id"), body.get("params") or {}
            self.methods.append(method)
            if method == "initialize":
                session = f"s{len(self.sessions) + len(self.methods)}"
                self.sessions.add(session)
                result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}}}
                response = self.reply(request_id, result | {"serverInfo": {"name": "t", "version": "1"}})
                response.headers["Mcp-Session-Id"] = session
                return response
            if request.headers.get("mcp-session-id") not in self.sessions:
                return Response(status_code=404)
            if request_id is None:
                return Response(status_code=202)
            if method == "tools/list":
                return self.reply(request_id, {"tools": self.tools})
            if self.status is not None:
                return Response(status_code=self.status)
            return self.reply(request_id, self.call(params["arguments"]))

        return app


class HostDispatch(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict[str, FastAPI]) -> None:
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._transports.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError("connection refused", request=request)
        return await transport.handle_async_request(request)


@dataclass
class Env:
    registry: Registry
    gateway: Gateway
    invoker: Invoker


@asynccontextmanager
async def environment(server: McpServer) -> AsyncIterator[Env]:
    async with httpx.AsyncClient(transport=HostDispatch({"mcp.test": server.app()})) as http:
        invoker = Invoker(
            http,
            secrets={"UEOA_SECRET_MCP": SECRET},
            token_provider=StaticTokenProvider(SECRET),
            entra_scopes=(SCOPE,),
        )
        registry = Registry({}, invoker, http, ALLOWED, entra_scopes=(SCOPE,))
        registry.register_capability(CONTRACT)
        yield Env(registry, Gateway(registry, invoker, Entitlements({"user": ["*"]})), invoker)


def _card(**params: Any) -> dict[str, Any]:
    ext = {
        "id": "upper-tool",
        "owner": "platform-tests",
        "kind": "tool",
        "implementation": "deterministic",
        "binding": "mcp",
        "implements": ["text.upper@1"],
        "effects": "read",
        "timeout_s": 5,
        "mcp": {"url": URL, "tool": "upper"},
    } | params
    raw = {"name": "Upper", "description": "MCP tool", "version": "1.0.0", "skills": [TOOL | {"id": "upper"}]}
    return with_extension(raw, ext)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize("mode", ["json", "sse"])
async def test_mcp_tool_registers_by_card_and_answers_through_the_gateway(mode: str) -> None:
    server = McpServer(mode=mode)
    async with environment(server) as env:
        impl = await env.registry.register_card(_card())
        assert (impl.source, impl.status) == ("submitted", ImplementationStatus.DRAFT)
        assert impl.conformant, impl.checks
        checks = {c.check: c.detail for c in impl.checks}
        assert checks["health"] == f"MCP tool 'upper' listed by {URL}"
        assert "mcp_tool_schema" in checks and "contract_sample[text.upper#0]" in checks
        env.registry.approve(impl.key)
        record = await env.gateway.invoke(USER, "text.upper", {"text": "hi"})
    assert (record.status, record.output) == ("succeeded", {"text": "HI"})
    assert next(s for s in record.steps if s.name == "invoke").detail == "mcp call to upper-tool@1.0.0"
    assert server.methods.count("initialize") == 1  # one session for the Invoker lifetime
    call = server.requests[-1]
    assert call["accept"] == "application/json, text/event-stream"
    assert call["mcp-protocol-version"] == "2025-06-18"
    assert call["mcp-session-id"] in server.sessions


async def test_expired_session_is_reinitialised_once() -> None:
    server = McpServer()
    async with environment(server) as env:
        impl = await env.registry.register_card(_card())
        server.sessions.clear()
        assert await env.invoker.invoke(impl.card, "text.upper", {"text": "again"}) == {"text": "AGAIN"}
    assert server.methods.count("initialize") == 2
    assert server.methods[-4:] == ["tools/call", "initialize", "notifications/initialized", "tools/call"]


@pytest.mark.parametrize(
    ("output", "expected"), [("json_text", {"text": "HI"}), ("plain", {"text": "HI"}), ("structured", {"text": "HI"})]
)
async def test_output_from_structured_or_text_content(output: str, expected: dict[str, Any]) -> None:
    async with environment(McpServer(output=output)) as env:
        card = (await env.registry.register_card(_card())).card
        assert await env.invoker.invoke(card, "text.upper", {"text": "hi"}) == expected


async def test_tool_error_is_a_remote_error_through_the_gateway() -> None:
    async with environment(McpServer()) as env:
        impl = await env.registry.register_card(_card())
        env.registry.approve(impl.key)
        with pytest.raises(GatewayFailedError) as failed:
            await env.gateway.invoke(USER, "text.upper", {"text": "fail"})
    assert failed.value.code == "remote_error"
    assert failed.value.record.error == "remote_error: upper-tool@1.0.0: tool exploded"


@pytest.mark.parametrize(("status", "code"), [(429, "rate_limited"), (503, "unavailable"), (403, "remote_error")])
async def test_http_status_mapping(status: int, code: str) -> None:
    server = McpServer()
    async with environment(server) as env:
        card = (await env.registry.register_card(_card())).card
        server.status = status
        with pytest.raises(InvocationError) as failed:
            await env.invoker.invoke(card, "text.upper", {"text": "hi"})
    assert failed.value.code == code


async def test_unreachable_server_fails_health_and_cannot_be_approved() -> None:
    async with environment(McpServer()) as env:
        impl = await env.registry.register_card(_card(mcp={"url": "http://other.test/mcp", "tool": "upper"}))
        health = next(c for c in impl.checks if c.check == "health")
        assert not health.passed and health.detail.startswith("MCP tools/list: unavailable")
        with pytest.raises(RegistryError, match="failing conformance"):
            env.registry.approve(impl.key)


async def test_tool_not_listed_fails_health() -> None:
    async with environment(McpServer(tools=[TOOL | {"name": "other"}])) as env:
        impl = await env.registry.register_card(_card())
    health = next(c for c in impl.checks if c.check == "health")
    assert not health.passed and "does not list tool 'upper'" in health.detail


async def test_tool_requiring_fields_outside_the_contract_fails_conformance() -> None:
    schema = {"type": "object", "required": ["text", "locale"], "properties": {"text": {}, "locale": {}}}
    async with environment(McpServer(tools=[TOOL | {"inputSchema": schema}])) as env:
        impl = await env.registry.register_card(_card())
    check = next(c for c in impl.checks if c.check == "mcp_tool_schema")
    assert not check.passed and check.detail == "text.upper: tool requires ['locale'], not in the input schema"


async def test_secret_auth_is_sent_as_headers() -> None:
    server = McpServer(token=SECRET)
    async with environment(server) as env:
        impl = await env.registry.register_card(_card(auth={"secret_ref": "UEOA_SECRET_MCP"}))
        assert impl.conformant, impl.checks
    assert all(h["authorization"] == f"Bearer {SECRET}" for h in server.requests)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mcp": {"url": "http://evil.example/mcp", "tool": "upper"}}, "allow-list"),
        (
            {"mcp": {"url": "http://x.wild.test/mcp", "tool": "upper"}, "auth": {"scheme": "entra", "scope": SCOPE}},
            "exact",
        ),
        (
            {"health_url": "http://other.test/healthz", "auth": {"scheme": "entra", "scope": SCOPE}},
            "must stay on host 'mcp.test'",
        ),
        ({"mcp": None}, "mcp settings are required"),
        ({"binding": "a2a"}, "only allowed with, binding 'mcp'"),
    ],
)
async def test_card_rules_apply_to_the_mcp_url(params: dict[str, Any], message: str) -> None:
    async with environment(McpServer()) as env:
        with pytest.raises(RegistryError, match=message):
            await env.registry.register_card(_card(**params))


async def test_entra_mcp_tool_on_its_own_host_gets_a_bearer() -> None:
    server = McpServer(token=SECRET)
    async with environment(server) as env:
        impl = await env.registry.register_card(_card(auth={"scheme": "entra", "scope": SCOPE}))
    assert impl.conformant, impl.checks
