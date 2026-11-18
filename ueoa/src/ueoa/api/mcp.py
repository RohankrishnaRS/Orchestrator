"""MCP gateway: entitled, approved capabilities as MCP tools over Streamable HTTP (plain JSON responses, no SSE).

No `from __future__ import annotations` here: FastAPI must evaluate `Depends(get_principal)` on a closure variable.
"""

import contextlib
import json
from collections.abc import Callable, Iterable
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection

from ueoa import __version__
from ueoa.contracts import NodeType
from ueoa.gateway.service import GatewayDeniedError, GatewayFailedError, InvocationRecord, Principal
from ueoa.orchestrator.planner import PlanResult
from ueoa.orchestrator.runner import RunError

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2025-11-25")
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
LOCAL_ORIGIN_HOSTS = frozenset({"localhost", "127.0.0.1"})
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}
MAX_WAIT_S = 300.0
PLAN_TOOL = "ueoa_plan"
RUN_TOOL = "ueoa_run_goal"

_GOAL_PROPS: dict[str, Any] = {
    "goal": {"type": "string", "description": "What to achieve, in plain language."},
    "inputs": {"type": "object", "description": "Named run inputs the workflow may bind to."},
}
META_TOOLS: list[dict[str, Any]] = [
    {
        "name": PLAN_TOOL,
        "description": "Plan a workflow for a goal from the capabilities you may use. Returns the plan; runs nothing.",
        "inputSchema": {"type": "object", "required": ["goal"], "properties": _GOAL_PROPS},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": RUN_TOOL,
        "description": "Plan a workflow for a goal, run it through the gateway and return the run (waits <= wait_s).",
        "inputSchema": {
            "type": "object",
            "required": ["goal"],
            "properties": {**_GOAL_PROPS, "wait_s": {"type": "number", "minimum": 0, "maximum": MAX_WAIT_S}},
        },
        "annotations": {"readOnlyHint": True},
    },
]


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def tool_name(capability_id: str) -> str:
    return capability_id.replace(".", "__")


def _origin_key(origin: str) -> tuple[str, str, int] | None:
    try:
        parts = urlsplit(origin.strip().lower())
        port = parts.port or DEFAULT_PORTS.get(parts.scheme)
    except ValueError:
        return None
    if not parts.hostname or port is None:
        return None
    return parts.scheme, parts.hostname, port


def origin_allowed(conn: HTTPConnection, allowed_origins: Iterable[str] = ()) -> bool:
    """No Origin (non-browser client), an exact configured origin, or else a local origin on this server's port."""
    origin = conn.headers.get("origin")
    if origin is None:
        return True
    key = _origin_key(origin)
    if key is None:
        return False
    allowed = {_origin_key(item) for item in allowed_origins} - {None}
    if allowed:
        return key in allowed
    own_port = conn.url.port or DEFAULT_PORTS.get(conn.url.scheme)
    return key[1] in LOCAL_ORIGIN_HOSTS and key[2] == own_port


def _rpc_error(request_id: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    body = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    return JSONResponse(body, status_code=status)


def _result(value: Any, *, error: bool = False, structured: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": json.dumps(value, default=str)}], "isError": error}
    if structured and isinstance(value, dict):
        result["structuredContent"] = value
    return result


def _gateway_error(record: InvocationRecord) -> dict[str, Any]:
    steps = "\n".join(f"{step.name}: {step.outcome} - {step.detail}" for step in record.steps)
    text = f"{record.status}: {record.error}\n{steps}"
    return {"content": [{"type": "text", "text": text}], "isError": True}


def create_mcp_router(
    get_services: Callable[[], Any], get_principal: Callable[..., Principal], *, allowed_origins: Iterable[str] = ()
) -> APIRouter:
    router = APIRouter(tags=["mcp"])
    allowed_origins = tuple(allowed_origins)

    def has_meta(services: Any) -> bool:
        return getattr(services, "planner", None) is not None and getattr(services, "runs", None) is not None

    def list_tools(services: Any, principal: Principal) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for capability_id in services.gateway.available(principal):
            contract = services.registry.capabilities[capability_id]
            description = contract.description
            if contract.selection_doc is not None:
                description += " Use when: " + contract.selection_doc.use_when
            tool: dict[str, Any] = {
                "name": tool_name(capability_id),
                "description": description,
                "inputSchema": contract.input_schema,
                "annotations": {"readOnlyHint": True},
            }
            if contract.output_schema.get("type") == "object":
                tool["outputSchema"] = contract.output_schema
            tools.append(tool)
        return tools + (META_TOOLS if has_meta(services) else [])

    async def plan(services: Any, principal: Principal, arguments: dict[str, Any]) -> tuple[PlanResult, dict[str, Any]]:
        goal, inputs = arguments.get("goal"), arguments.get("inputs") or {}
        if not isinstance(goal, str) or not goal.strip() or not isinstance(inputs, dict):
            raise _RpcError(-32602, "arguments need a non-empty 'goal' string and an optional 'inputs' object")
        available = services.gateway.available(principal)
        result = await services.planner.plan(goal, inputs, services.registry.capabilities, available)
        return result, inputs  # pyright: ignore[reportUnknownVariableType]

    async def call_meta(services: Any, principal: Principal, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        wait_s = arguments.get("wait_s", 60)
        if not isinstance(wait_s, int | float) or isinstance(wait_s, bool):
            raise _RpcError(-32602, "'wait_s' must be a number")
        result, inputs = await plan(services, principal, arguments)
        view = result.view()
        if name == PLAN_TOOL or not result.ok:
            return _result(view, error=not result.ok, structured=False)
        spec = result.spec
        available = set(services.gateway.available(principal))
        nodes = [n for n in spec.nodes if n.type is NodeType.CAPABILITY and n.capability]
        blocked = sorted({n.capability for n in nodes if n.capability not in available})
        if blocked:
            return _result(
                {"error": f"not allowed or not available: {blocked}", "plan": view}, error=True, structured=False
            )
        node_timeout = getattr(services, "node_timeout", None)
        if node_timeout is not None and not getattr(services, "runner_applies_node_timeout", True):
            for node in nodes:
                if (minimum := node_timeout(node.capability)) is not None:
                    node.timeout_s = max(node.timeout_s or 0, int(minimum))
        try:
            run = await services.runs.start(spec, inputs, principal, plan=view)
        except RunError as exc:
            return _result({"error": str(exc), "plan": view}, error=True, structured=False)
        with contextlib.suppress(TimeoutError):  # still running: the caller gets the current view
            await services.runs.wait(run.id, timeout=min(max(float(wait_s), 0.0), MAX_WAIT_S))
        return _result(run.view(), structured=False)

    async def call_tool(services: Any, principal: Principal, params: dict[str, Any]) -> dict[str, Any]:
        name, arguments = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise _RpcError(-32602, "tools/call needs a 'name' string and an 'arguments' object")
        if name in (PLAN_TOOL, RUN_TOOL) and has_meta(services):
            return await call_meta(services, principal, name, arguments)  # pyright: ignore[reportUnknownArgumentType]
        capability_id = {tool_name(cid): cid for cid in services.gateway.available(principal)}.get(name)
        if capability_id is None:
            raise _RpcError(-32602, f"unknown tool '{name}'")
        try:
            record = await services.gateway.invoke(principal, capability_id, arguments)
        except (GatewayDeniedError, GatewayFailedError) as exc:
            return _gateway_error(exc.record)
        return _result(record.output)

    async def dispatch(services: Any, principal: Principal, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            requested = params.get("protocolVersion")
            return {
                "protocolVersion": requested if requested in PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "ueoa-gateway", "version": __version__},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": list_tools(services, principal)}
        if method == "tools/call":
            return await call_tool(services, principal, params)
        raise _RpcError(-32601, f"method '{method}' not found")

    @router.post("/mcp")
    async def mcp(request: Request, principal: Annotated[Principal, Depends(get_principal)]) -> Response:  # pyright: ignore[reportUnusedFunction]
        if not origin_allowed(request, allowed_origins):
            return Response(status_code=403)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return Response(status_code=415)
        try:
            body = json.loads(await request.body())
        except ValueError:
            return _rpc_error(None, -32700, "parse error", 400)
        if isinstance(body, list):
            return _rpc_error(None, -32600, "batch requests are not supported", 400)
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            return _rpc_error(None, -32600, "invalid JSON-RPC 2.0 message", 400)
        if "id" not in body:  # notifications and client responses
            return Response(status_code=202)
        request_id, method, params = body["id"], body.get("method"), body.get("params") or {}
        if not isinstance(method, str) or not isinstance(params, dict):
            return _rpc_error(request_id, -32600, "invalid request")
        try:
            result = await dispatch(get_services(), principal, method, params)  # pyright: ignore[reportUnknownArgumentType]
        except _RpcError as exc:
            return _rpc_error(request_id, exc.code, str(exc))
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

    @router.get("/mcp")
    def mcp_stream() -> Response:  # pyright: ignore[reportUnusedFunction]
        return Response(status_code=405, headers={"Allow": "POST"})

    return router
