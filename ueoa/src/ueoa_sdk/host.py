"""Generic A2A agent host: publishes the agent card, a health endpoint and the A2A JSON-RPC endpoint.

Accepts A2A 1.0 (`SendMessage`) and 0.3 (`message/send`) requests and answers in the matching shape.
Input is the first structured `data` part (or a JSON text part); output is returned as a task artifact.
"""

from __future__ import annotations

import copy
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ueoa.a2a import InvocationContext, SkillHandler, handle_jsonrpc
from ueoa.registry.models import UEOA_EXTENSION_URI

__all__ = ["InvocationContext", "SkillHandler", "build_card", "create_agent_app", "ueoa_extension"]


def ueoa_extension(card: dict[str, Any]) -> dict[str, Any]:
    for ext in card["capabilities"]["extensions"]:
        if ext["uri"] == UEOA_EXTENSION_URI:
            return ext["params"]
    raise KeyError("card has no UEOA extension")


def build_card(base_url: str, card: dict[str, Any]) -> dict[str, Any]:
    built = copy.deepcopy(card)
    built["supportedInterfaces"] = [{"url": f"{base_url}/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]
    ueoa_extension(built)["health_url"] = f"{base_url}/healthz"
    return built


def create_agent_app(card: dict[str, Any], handlers: dict[str, SkillHandler], base_url: str) -> FastAPI:
    app = FastAPI(title=card["name"], docs_url=None, redoc_url=None)
    published = build_card(base_url, card)

    @app.get("/.well-known/agent-card.json")
    def agent_card() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return published

    @app.get("/healthz")
    def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.post("/a2a")
    async def a2a(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(await handle_jsonrpc(await request.body(), handlers))

    return app
