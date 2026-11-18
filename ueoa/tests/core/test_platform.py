"""The platform starts empty and learns everything at runtime: contracts, agents, invocation. No domain content."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ueoa.api.app import Services, Settings, build_services, create_app
from ueoa.gateway.service import GatewayDeniedError, Principal
from ueoa.registry.models import UEOA_EXTENSION_URI, ImplementationStatus
from ueoa.registry.service import RegistryError
from ueoa_sdk.host import InvocationContext, create_agent_app

pytestmark = pytest.mark.anyio

USER = Principal("demo-user", "user")
ECHO_CONTRACT: dict[str, Any] = {
    "id": "text.echo",
    "version": "1.0.0",
    "description": "Return the input text unchanged.",
    "effects": "read",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
    },
    "output_schema": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
    "samples": [{"text": "hello"}],
}
WRITE_CONTRACT: dict[str, Any] = {
    **ECHO_CONTRACT,
    "id": "record.write",
    "effects": "write",
    "description": "Write a record externally.",
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


async def _echo(payload: dict[str, Any], _ctx: InvocationContext) -> dict[str, Any]:
    return {"text": payload["text"]}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def services() -> AsyncIterator[Services]:
    agent = create_agent_app(ECHO_CARD, {"text.echo": _echo}, "http://echo.test")
    transport = httpx.ASGITransport(app=agent)
    async with httpx.AsyncClient(transport=transport) as http:
        yield await build_services(Settings(catalog_dir=None, allowed_hosts=("echo.test",)), http)


async def test_platform_starts_empty(services: Services) -> None:
    assert services.registry.capabilities == {}
    assert services.registry.implementations() == []
    assert services.entitlements.roles() == ["admin", "user"]


async def test_register_contract_then_agent_then_invoke(services: Services) -> None:
    services.registry.register_capability(ECHO_CONTRACT)
    impl = await services.registry.register_from_url("http://echo.test")
    assert impl.conformant, [c for c in impl.checks if not c.passed]
    assert impl.status is ImplementationStatus.DRAFT
    services.registry.approve(impl.key)
    record = await services.gateway.invoke(USER, "text.echo", {"text": "ping"})
    assert (record.implementation, record.output) == ("echo-agent@1.0.0", {"text": "ping"})


async def test_agent_for_unknown_contract_fails_conformance(services: Services) -> None:
    impl = await services.registry.register_from_url("http://echo.test")
    assert "implements_known_capability" in {c.check for c in impl.checks if not c.passed}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(input_schema={"type": "nope"}), "not a valid JSON Schema"),
        (lambda c: c.update(samples=[{"wrong": 1}]), "does not match input_schema"),
        (lambda c: c.update(id="Bad Id"), "invalid capability contract"),
    ],
)
async def test_bad_contracts_are_rejected(services: Services, mutate: Any, message: str) -> None:
    contract = dict(ECHO_CONTRACT)
    mutate(contract)
    with pytest.raises(RegistryError, match=message):
        services.registry.register_capability(contract)


async def test_duplicate_contract_is_rejected(services: Services) -> None:
    services.registry.register_capability(ECHO_CONTRACT)
    with pytest.raises(RegistryError, match="already registered"):
        services.registry.register_capability(ECHO_CONTRACT)


async def test_write_contract_is_denied_by_gateway(services: Services) -> None:
    services.registry.register_capability(WRITE_CONTRACT)
    with pytest.raises(GatewayDeniedError) as denied:
        await services.gateway.invoke(USER, "record.write", {"text": "x"})
    assert denied.value.record.steps[-1].name == "effect"


async def test_local_tools_need_an_allow_listed_package(services: Services) -> None:
    services.registry.register_capability(ECHO_CONTRACT)
    seed = {**ECHO_CARD, "url": "local"}
    seed["capabilities"] = {
        "extensions": [
            {
                "uri": UEOA_EXTENSION_URI,
                "params": {
                    **ECHO_CARD["capabilities"]["extensions"][0]["params"],
                    "binding": "local",
                    "entrypoint": "os:getcwd",
                },
            }
        ]
    }
    with pytest.raises(RegistryError, match="failed conformance"):
        await services.registry.register_seed(seed)


def test_api_capability_registration_is_admin_only() -> None:
    with TestClient(create_app(Settings(catalog_dir=None))) as client:
        assert client.get("/api/v1/capabilities").json() == []
        assert client.get("/api/v1/roles").json() == ["admin", "user"]
        assert client.post("/api/v1/capabilities", json=ECHO_CONTRACT).status_code == 403
        created = client.post("/api/v1/capabilities", json=ECHO_CONTRACT, headers={"X-Demo-Role": "admin"})
        assert created.status_code == 201
        assert [c["id"] for c in client.get("/api/v1/capabilities").json()] == ["text.echo"]
