"""Managed catalog of remote MCP server endpoints."""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import parse_qsl, urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator

from ueoa.gateway.service import Principal
from ueoa.store import StoreError

KIND = "mcp-servers"


class McpServerRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    url: AnyHttpUrl
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        parts = urlsplit(str(value))
        if parts.username or parts.password:
            raise ValueError("credentials must not be embedded in the MCP server URL")
        if parts.fragment:
            raise ValueError("MCP server URL must not include a fragment")
        credential_names = ("auth", "token", "key", "secret", "password", "credential")
        if any(
            name.lower() == credential or name.lower().endswith(credential)
            for name, _ in parse_qsl(parts.query)
            for credential in credential_names
        ):
            raise ValueError("credentials must not be embedded in the MCP server URL")
        return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _server_id(name: str) -> str:
    server_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:81].rstrip("-")
    if not server_id:
        raise HTTPException(422, "name must include at least one letter or number")
    return server_id


def create_router(
    services: Callable[[], Any],
    principal: Callable[..., Principal],
    admin: Callable[..., Principal],
    _settings: Any,
) -> APIRouter:
    router = APIRouter()
    Caller = Annotated[Principal, Depends(principal)]
    Admin = Annotated[Principal, Depends(admin)]

    @router.get("/api/v1/mcp-servers")
    def list_servers(_: Caller) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return services().store.list(KIND)

    @router.post("/api/v1/mcp-servers", status_code=201)
    def register_server(body: McpServerRequest, who: Admin) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        store = services().store
        server_id = _server_id(body.name)
        try:
            if store.get(KIND, server_id) is not None:
                raise HTTPException(409, f"MCP server '{server_id}' already exists")
            server = {
                "id": server_id,
                "name": body.name,
                "url": str(body.url),
                "description": body.description.strip(),
                "transport": "streamable-http",
                "owner": who.name,
                "registered_at": _now(),
            }
            store.put(KIND, server_id, server)
            return server
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.delete("/api/v1/mcp-servers/{server_id}")
    def delete_server(server_id: str, _: Admin) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        store = services().store
        try:
            server = store.get(KIND, server_id)
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc
        if server is None:
            raise HTTPException(404, f"unknown MCP server '{server_id}'")
        store.delete(KIND, server_id)
        return {"id": server_id, "deleted": True}

    return router
