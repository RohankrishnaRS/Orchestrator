"""Gateway error codes end to end (HTTP status, transport, JSON-RPC codes from the SDK host) and Gateway.available."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from ueoa.gateway.invokers import Invoker
from ueoa.gateway.service import Entitlements, Gateway, GatewayFailedError, Principal
from ueoa.llm import LLMError, LLMRateLimitedError, LLMUnavailableError
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa.registry.service import Registry
from ueoa_sdk.host import InvocationContext, build_card, create_agent_app

pytestmark = pytest.mark.anyio

BASE = "http://agent.test"
USER = Principal("tester", "user")
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}


def _contract(capability_id: str, effects: str = "read") -> dict[str, Any]:
    return {
        "id": capability_id,
        "version": "1.0.0",
        "description": f"Neutral {capability_id}.",
        "effects": effects,
        "input_schema": SCHEMA,
        "output_schema": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
        "samples": [{"text": "hello"}],
    }


def _card(agent_id: str, implements: list[str], effects: str = "read") -> dict[str, Any]:
    params = {
        "id": agent_id,
        "owner": "platform-tests",
        "kind": "agent",
        "implementation": "deterministic",
        "binding": "a2a",
        "implements": implements,
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


@dataclass
class Switch:
    status: int | None = None
    transport_error: type[httpx.TransportError] | None = None
    skill_error: BaseException | None = None
    output: dict[str, Any] | None = None


@dataclass
class Env:
    registry: Registry
    gateway: Gateway
    switch: Switch


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    switch = Switch()

    async def skill(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
        if switch.skill_error is not None:
            raise switch.skill_error
        return switch.output or {"text": payload["text"]}

    handlers = {"text.echo": skill, "text.draft": skill, "record.write": skill}
    asgi = httpx.ASGITransport(app=create_agent_app(_card("echo", ["text.echo@1"]), handlers, BASE))

    async def route(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a2a":
            if switch.status is not None:
                return httpx.Response(switch.status, text="hidden detail")
            if switch.transport_error is not None:
                raise switch.transport_error("boom", request=request)
        return await asgi.handle_async_request(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as http:
        invoker = Invoker(http)
        registry = Registry({}, invoker, http, ("agent.test",))
        for capability_id, effects in (("text.echo", "read"), ("text.draft", "read"), ("record.write", "write")):
            registry.register_capability(_contract(capability_id, effects))
        echo = await registry.register_from_url(BASE)
        registry.approve(echo.key)
        await registry.register_card(build_card(BASE, _card("drafter", ["text.draft@1"])))  # left in draft
        writer = await registry.register_card(build_card(BASE, _card("writer", ["record.write@1"], "write")))
        registry.approve(writer.key)
        entitlements = Entitlements({"user": ["text.*", "record.*"], "narrow": ["text.draft"], "admin": ["*"]})
        yield Env(registry, Gateway(registry, invoker, entitlements), switch)


@pytest.mark.parametrize(
    ("setting", "value", "code"),
    [
        ("status", 429, "rate_limited"),
        ("status", 503, "unavailable"),
        ("status", 404, "remote_error"),
        ("transport_error", httpx.ConnectError, "unavailable"),
        ("transport_error", httpx.ReadTimeout, "timeout"),
        ("skill_error", LLMRateLimitedError("HTTP 429"), "rate_limited"),
        ("skill_error", LLMUnavailableError("HTTP 503"), "unavailable"),
        ("skill_error", TimeoutError(), "timeout"),
        ("skill_error", LLMError("bad shape"), "remote_error"),
    ],
)
async def test_failures_carry_retryable_codes(env: Env, setting: str, value: Any, code: str) -> None:
    setattr(env.switch, setting, value)
    with pytest.raises(GatewayFailedError) as failed:
        await env.gateway.invoke(USER, "text.echo", {"text": "x"})
    assert failed.value.code == code
    record = failed.value.record
    assert record.status == "failed" and record.error is not None and record.error.startswith(f"{code}: ")
    assert "hidden detail" not in record.error
    assert record.steps[-1].name == "invoke" and record.steps[-1].outcome == "error"


async def test_output_schema_failure_is_bad_output(env: Env) -> None:
    env.switch.output = {"wrong": 1}
    with pytest.raises(GatewayFailedError) as failed:
        await env.gateway.invoke(USER, "text.echo", {"text": "x"})
    assert failed.value.code == "bad_output"
    assert failed.value.record.error == "bad_output: output schema: 'text' is a required property"
    assert failed.value.record.steps[-1].name == "output_schema"


async def test_success_is_unchanged(env: Env) -> None:
    record = await env.gateway.invoke(USER, "text.echo", {"text": "x"})
    assert (record.status, record.output) == ("succeeded", {"text": "x"})


async def test_available_filters_entitlement_approval_and_writes(env: Env) -> None:
    assert env.gateway.available(USER) == ["text.echo"]
    assert env.gateway.available(USER, include_writes=True) == ["record.write", "text.echo"]
    assert env.gateway.available(Principal("n", "narrow"), include_writes=True) == []
    assert env.gateway.available(Principal("x", "unknown")) == []
    env.registry.approve("drafter@1.0.0")
    assert env.gateway.available(Principal("n", "narrow")) == ["text.draft"]
