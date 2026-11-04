"""Entra-authenticated A2A agents: card at a custom path behind a bearer, text-only parts, task polling."""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from ueoa.gateway.auth import EntraTokenProvider, StaticTokenProvider, TokenError
from ueoa.gateway.invokers import InvocationError, Invoker
from ueoa.gateway.service import Entitlements, Gateway, Principal
from ueoa.registry.models import with_extension
from ueoa.registry.service import Registry, RegistryError
from ueoa_sdk.host import InvocationContext, build_card, create_agent_app

pytestmark = pytest.mark.anyio

TOKEN = "test-token"  # noqa: S105 - a fake bearer accepted only by the in-test agent
AGENT = "http://agent.test/svc"
SDK_AGENT = "http://sdk.test"
SCOPE = "api://test/.default"
USER = Principal("tester", "user")
SKILL = {"id": "s", "name": "S", "description": "s"}
TEXT_SCHEMA: dict[str, Any] = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
CONTRACT: dict[str, Any] = {
    "id": "text.echo",
    "version": "1.0.0",
    "description": "Neutral echo.",
    "effects": "read",
    "input_schema": TEXT_SCHEMA,
    "output_schema": TEXT_SCHEMA,
    "samples": [{"text": "hello"}],
}


def _overlay(output: str = "json") -> dict[str, Any]:
    return {
        "id": "echo-text",
        "owner": "platform-tests",
        "kind": "agent",
        "implementation": "deterministic",
        "binding": "a2a",
        "implements": ["text.echo@1"],
        "effects": "read",
        "timeout_s": 5,
        "auth": {"scheme": "entra", "scope": SCOPE},
        "a2a": {"message": "text", "output": output, "text_field": "text", "card_path": "card"},
    }


@dataclass
class TextAgent:
    """A2A agent that, like hosted agent services, requires a bearer, accepts text parts only and may answer later."""

    version: str = "1.0"
    mode: str = "ok"  # ok | slow | fail | plain
    calls: list[str] = field(default_factory=lambda: [])
    replies: dict[str, str] = field(default_factory=lambda: {})

    def state(self, name: str) -> str:
        return f"TASK_STATE_{name.upper()}" if self.version == "1.0" else name

    def task(self, task_id: str, state: str, text: str | None = None, status_text: str = "") -> dict[str, Any]:
        def part(value: str) -> dict[str, Any]:
            return {"text": value} if self.version == "1.0" else {"kind": "text", "text": value}

        status: dict[str, Any] = {"state": self.state(state)}
        if status_text:
            status["message"] = {"role": "ROLE_AGENT", "parts": [part(status_text)]}
        task: dict[str, Any] = {"id": task_id, "contextId": task_id, "status": status}
        if text is not None:
            task["artifacts"] = [{"artifactId": "a1", "parts": [part(text)]}]
        if self.version != "1.0":
            task["kind"] = "task"
        return task

    def app(self) -> FastAPI:
        app = FastAPI()

        def authorize(authorization: str | None) -> None:
            if authorization != f"Bearer {TOKEN}":
                raise HTTPException(401, "bearer required")

        @app.get("/svc/card")
        def card(authorization: str | None = Header(default=None)) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
            authorize(authorization)
            interface = {"url": AGENT, "protocolBinding": "JSONRPC", "protocolVersion": self.version}
            return {
                "name": "Text agent",
                "description": "Echoes text in upper case.",
                "version": "1.0.0",
                "supportedInterfaces": [interface],
                "defaultInputModes": ["text/plain"],
                "skills": [{"id": "echo", "name": "Echo", "description": "Echo"}],
            }

        @app.post("/svc")
        async def rpc(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
            authorize(authorization)
            body = await request.json()
            method, rid = body["method"], body["id"]
            self.calls.append(method)
            send, get = ("SendMessage", "GetTask") if self.version == "1.0" else ("message/send", "tasks/get")
            if self.version == "1.0" and request.headers.get("A2A-Version") != "1.0":
                return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no method"}})
            if method == get:
                task_id = body["params"]["id"]
                if self.mode == "fail":
                    result = self.task(task_id, "failed", status_text="quota exhausted")
                else:
                    result = self.task(task_id, "completed", self.replies[task_id])
                return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result})
            parts = body["params"]["message"]["parts"]
            if method != send or any("data" in p or not isinstance(p.get("text"), str) for p in parts):
                return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "text only"}})
            upper = json.loads("".join(p["text"] for p in parts))["text"].upper()
            task_id = str(uuid.uuid4())
            self.replies[task_id] = upper if self.mode == "plain" else json.dumps({"text": upper})
            if self.mode in {"slow", "fail"}:
                result = self.task(task_id, "working")
            else:
                result = self.task(task_id, "completed", self.replies[task_id])
            return JSONResponse(
                {"jsonrpc": "2.0", "id": rid, "result": {"task": result} if self.version == "1.0" else result}
            )

        return app


class HostDispatch(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict[str, FastAPI]) -> None:
        self.requests: list[httpx.Request] = []
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._transports[request.url.host].handle_async_request(request)


@dataclass
class Env:
    registry: Registry
    gateway: Gateway
    invoker: Invoker
    transport: HostDispatch


def _sdk_card() -> dict[str, Any]:
    params = {k: v for k, v in _overlay().items() if k not in {"auth", "a2a"}} | {"id": "echo-data"}
    raw = {"name": "Data agent", "description": "d", "version": "1.0.0", "skills": [SKILL]}
    return with_extension(raw, params)


async def _sdk_echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


@asynccontextmanager
async def environment(agent: TextAgent, token: str | None = TOKEN) -> AsyncIterator[Env]:
    sdk_app = create_agent_app(_sdk_card(), {"text.echo": _sdk_echo}, SDK_AGENT)
    transport = HostDispatch({"agent.test": agent.app(), "sdk.test": sdk_app})
    provider = StaticTokenProvider(token) if token else None
    async with httpx.AsyncClient(transport=transport) as http:
        invoker = Invoker(http, poll_interval_s=0.01, token_provider=provider, entra_scopes=(SCOPE,))
        registry = Registry({}, invoker, http, ("agent.test", "sdk.test"), entra_scopes=(SCOPE,))
        registry.register_capability(CONTRACT)
        gateway = Gateway(registry, invoker, Entitlements({"user": ["*"]}))
        yield Env(registry, gateway, invoker, transport)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize("version", ["1.0", "0.3"])
@pytest.mark.parametrize("mode", ["ok", "slow"])
async def test_entra_text_agent_registers_and_answers_through_the_gateway(version: str, mode: str) -> None:
    agent = TextAgent(version=version, mode=mode)
    async with environment(agent) as env:
        impl = await env.registry.register_from_url(AGENT, overlay=_overlay())
        assert impl.source == f"{AGENT}/card"
        assert impl.conformant, impl.checks
        env.registry.approve(impl.key)
        record = await env.gateway.invoke(USER, "text.echo", {"text": "hi"})
    assert (record.status, record.output) == ("succeeded", {"text": "HI"})
    send, get = ("SendMessage", "GetTask") if version == "1.0" else ("message/send", "tasks/get")
    assert agent.calls == ([send, get] * 2 if mode == "slow" else [send] * 2)
    assert all(r.headers["Authorization"] == f"Bearer {TOKEN}" for r in env.transport.requests)
    if version == "1.0":
        assert all(r.headers.get("A2A-Version") == "1.0" for r in env.transport.requests if r.method == "POST")


async def test_failed_task_after_polling_is_remote_error_with_status_message() -> None:
    agent = TextAgent()
    async with environment(agent) as env:
        impl = await env.registry.register_from_url(AGENT, overlay=_overlay())
        assert impl.conformant, impl.checks
        agent.mode = "fail"
        with pytest.raises(InvocationError) as failed:
            await env.invoker.invoke(impl.card, "text.echo", {"text": "hi"})
    assert failed.value.code == "remote_error"
    assert str(failed.value) == "echo-text@1.0.0 task ended in state 'failed': quota exhausted"


async def test_text_output_mode_wraps_plain_text() -> None:
    async with environment(TextAgent(mode="plain")) as env:
        impl = await env.registry.register_from_url(AGENT, overlay=_overlay(output="text"))
        assert impl.conformant, impl.checks
        assert await env.invoker.invoke(impl.card, "text.echo", {"text": "hi"}) == {"text": "HI"}


async def test_missing_token_provider_is_not_allowed() -> None:
    async with environment(TextAgent(), token=None) as env:
        with pytest.raises(RegistryError, match="no token provider is configured"):
            await env.registry.register_from_url(AGENT, overlay=_overlay())
        raw = {
            "name": "Text agent",
            "description": "d",
            "version": "1.0.0",
            "supportedInterfaces": [{"url": AGENT, "protocolVersion": "1.0"}],
            "skills": [SKILL],
        }
        impl = await env.registry.register_card(with_extension(raw, _overlay()))
        failed = [c for c in impl.checks if not c.passed]
        assert failed and failed[0].detail.startswith("not_allowed: ")
        with pytest.raises(InvocationError) as denied:
            await env.invoker.invoke(impl.card, "text.echo", {"text": "hi"})
    assert denied.value.code == "not_allowed"


async def test_wrong_token_is_rejected_at_card_fetch() -> None:
    async with environment(TextAgent(), token="wrong") as env:  # noqa: S106 - fake token
        with pytest.raises(RegistryError, match="HTTP 401"):
            await env.registry.register_from_url(AGENT, overlay=_overlay())


async def test_data_part_agents_are_unchanged() -> None:
    async with environment(TextAgent()) as env:
        impl = await env.registry.register_card(build_card(SDK_AGENT, _sdk_card()))
        assert impl.conformant, impl.checks
        env.registry.approve(impl.key)
        record = await env.gateway.invoke(USER, "text.echo", {"text": "hi"})
    assert record.output == {"text": "hi"}
    posts = [r for r in env.transport.requests if r.method == "POST"]
    assert posts and all("Authorization" not in r.headers for r in posts)
    assert json.loads(posts[-1].content)["params"]["message"]["parts"] == [
        {"data": {"text": "hi"}, "mediaType": "application/json"}
    ]


async def test_overlay_a2a_options_need_a2a_binding() -> None:
    async with environment(TextAgent()) as env:
        overlay = {**_overlay(), "binding": "copilot-studio"}
        with pytest.raises(RegistryError, match="invalid overlay"):
            await env.registry.register_from_url(AGENT, overlay=overlay)


@dataclass
class FakeAccessToken:
    token: str
    expires_on: int


class FakeCredential:
    def __init__(self, lifetime_s: int) -> None:
        self.lifetime_s = lifetime_s
        self.scopes: list[str] = []
        self.closed = False

    async def get_token(self, scope: str) -> FakeAccessToken:
        self.scopes.append(scope)
        return FakeAccessToken(f"tok-{len(self.scopes)}", int(time.time()) + self.lifetime_s)

    async def close(self) -> None:
        self.closed = True


async def test_entra_provider_caches_until_five_minutes_before_expiry() -> None:
    long_lived = FakeCredential(3600)
    provider = EntraTokenProvider(long_lived)
    assert [await provider.token("s1"), await provider.token("s1"), await provider.token("s2")] == [
        "tok-1",
        "tok-1",
        "tok-2",
    ]
    assert long_lived.scopes == ["s1", "s2"]
    await provider.close()
    assert long_lived.closed

    near_expiry = FakeCredential(200)
    provider = EntraTokenProvider(near_expiry)
    assert [await provider.token("s"), await provider.token("s")] == ["tok-1", "tok-2"]


async def test_entra_provider_without_azure_identity_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "azure.identity.aio", None)
    with pytest.raises(TokenError, match=r"pip install -e \.\[azure\]"):
        await EntraTokenProvider().token("api://test/.default")
