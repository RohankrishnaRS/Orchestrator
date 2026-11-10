# pyright: basic
"""Goals API: the versioned contract external clients (Microsoft 365 Copilot adapter) call to execute a goal.

POST /api/v1/goals            plan a goal with the planner and run it through the gateway (200 done / 202 running)
GET  /api/v1/goals/{run_id}   status, node progress and outputs of a goal run
Every response carries X-Correlation-Id (client-supplied or generated) in the header and the body, and every error
uses one shape: {code, message, correlation_id, details?}.
No `from __future__ import annotations` here: FastAPI must evaluate `Depends(...)` on closure-local functions.
"""

import importlib
import json
import logging
import re
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ueoa.api.app import Services, Settings, available_capabilities, routing_view
from ueoa.contracts import NodeType
from ueoa.gateway.service import Principal

logger = logging.getLogger(__name__)

RUNNING_STATES = frozenset({"queued", "pending", "running", "waiting_input", "waiting_approval"})
CORRELATION_RE = re.compile(r"^[\x21-\x7e]{1,128}$")  # printable ASCII, no spaces, <= 128 chars
STORE_KIND = "goal-runs"


class GoalRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    inputs: dict[str, Any] = Field(default_factory=dict)
    wait_seconds: int = Field(default=60, ge=0, le=300)


class GoalError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def _correlation_id(header: str | None) -> str:
    if header is None or not header.strip():
        return str(uuid.uuid4())
    value = header.strip()
    if not CORRELATION_RE.match(value):
        raise GoalError(422, "validation_error", "X-Correlation-Id must be 1-128 printable characters without spaces")
    return value


def _error_response(exc: GoalError, correlation_id: str) -> JSONResponse:
    body: dict[str, Any] = {"code": exc.code, "message": exc.message, "correlation_id": correlation_id}
    if exc.details is not None:
        body["details"] = exc.details
    return JSONResponse(body, status_code=exc.status, headers={"X-Correlation-Id": correlation_id})


def _markdown(outputs: Any) -> str:
    """outputs.report_markdown when present, else a compact Markdown rendering of the outputs."""
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if key.endswith("_markdown") and isinstance(value, str) and value.strip():
                return value
        lines = ["### Result", ""]
        for key, value in outputs.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > 1500:
                text = text[:1500] + " ..."
            lines.append(f"- **{key}**: {text}")
        return "\n".join(lines) + "\n"
    if outputs is None:
        return ""
    return f"```json\n{json.dumps(outputs, indent=2, ensure_ascii=False, default=str)[:4000]}\n```\n"


def _nodes(view: dict[str, Any]) -> list[dict[str, Any]]:
    raw = view.get("nodes")
    items: list[Any] = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "node": item.get("id") or item.get("node_id") or item.get("node"),
                "capability": item.get("capability"),
                "status": str(item.get("status", "")),
                "duration_ms": item.get("duration_ms") or item.get("elapsed_ms"),
                "items": len(item["items"]) if isinstance(item.get("items"), list) else None,
            }
        )
    return out


def _goal_view(run: Any, correlation_id: str, plan_summary: list[dict[str, Any]], routing: Any) -> dict[str, Any]:
    view: dict[str, Any] = run.view()
    status = str(run.status)
    outputs = view.get("outputs")
    return {
        "run_id": run.id,
        "correlation_id": correlation_id,
        "status": status,
        "finished": status not in RUNNING_STATES,
        "plan_summary": plan_summary,
        "routing": routing,
        "nodes": _nodes(view),
        "outputs": outputs,
        "result_markdown": _markdown(outputs) if status not in RUNNING_STATES else None,
        "links": {"run": f"/api/v1/runs/{run.id}", "goal": f"/api/v1/goals/{run.id}"},
    }


def _plan_summary(spec: Any) -> list[dict[str, Any]]:
    return [
        {"node": node.id, "capability": node.capability, "type": str(node.type)}
        for node in spec.nodes
    ]


def create_router(
    services: Callable[[], Services],
    principal: Callable[..., Principal],
    admin: Callable[..., Principal],
    settings: Settings,
) -> APIRouter:
    del admin, settings
    router = APIRouter(tags=["Goals"])
    Caller = Annotated[Principal, Depends(principal)]
    CorrelationHeader = Annotated[str | None, Header(alias="X-Correlation-Id")]
    run_error: type[Exception] = getattr(importlib.import_module("ueoa.api.app"), "RunError", Exception)

    def remember(run_id: str, record: dict[str, Any]) -> None:
        with suppress(Exception):
            services().store.put(STORE_KIND, run_id, record)

    def recall(run_id: str) -> dict[str, Any]:
        with suppress(Exception):
            found = services().store.get(STORE_KIND, run_id)
            if isinstance(found, dict):
                return found
        return {}

    @router.post(
        "/api/v1/goals",
        summary="Plan and execute a goal",
        description=(
            "Plans the goal with the platform planner, runs it through the gateway as the caller, and waits up to "
            "`wait_seconds`. Returns 200 when the run finished within the wait, 202 when it is still running "
            "(poll GET /api/v1/goals/{run_id}). Errors use {code, message, correlation_id, details?}."
        ),
        responses={
            200: {"description": "Run finished"},
            202: {"description": "Run still in progress"},
            403: {"description": "forbidden: capability not entitled or not available"},
            422: {"description": "validation_error: bad request or the planner produced no valid plan"},
            503: {"description": "planner_unavailable"},
        },
    )
    async def start_goal(  # pyright: ignore[reportUnusedFunction]
        body: GoalRequest, who: Caller, response: Response, x_correlation_id: CorrelationHeader = None
    ) -> Any:
        try:
            correlation_id = _correlation_id(x_correlation_id)
        except GoalError as exc:
            return _error_response(exc, str(uuid.uuid4()))
        response.headers["X-Correlation-Id"] = correlation_id
        try:
            svc = services()
            if svc.planner is None:
                raise GoalError(503, "planner_unavailable", "The planner is not configured on this server.")
            available = available_capabilities(svc, who)
            planned = await svc.planner.plan(body.goal, body.inputs, svc.registry.capabilities, available)
            if not planned.ok or planned.spec is None:
                raise GoalError(422, "validation_error", "The planner could not produce a valid plan.", planned.view())
            spec = planned.spec
            capability_nodes = [n for n in spec.nodes if n.type is NodeType.CAPABILITY and n.capability]
            blocked = sorted({n.capability for n in capability_nodes if n.capability not in available})
            if blocked:
                raise GoalError(403, "forbidden", "Not allowed or not available.", {"capabilities": blocked})
            if svc.node_timeout is not None:
                for node in capability_nodes:
                    if node.timeout_s is None or not svc.runner_applies_node_timeout:
                        minimum = int(svc.node_timeout(node.capability or "") or 0)
                        node.timeout_s = max(node.timeout_s or 0, minimum) or None
            routing = routing_view(svc, who, spec, available, planned.considered)
            meta: dict[str, Any] = {
                "goal": body.goal,
                "rationale": planned.rationale,
                "attempts": planned.attempts,
                "considered": planned.considered,
                "planned_by": "planner",
                "routing": routing,
                "correlation_id": correlation_id,
                "channel": "goals-api",
            }
            try:
                run = await svc.runs.start(spec, body.inputs, who, plan=meta)
            except run_error as exc:
                raise GoalError(422, "validation_error", str(exc)) from exc
            plan_summary = _plan_summary(spec)
            remember(run.id, {"correlation_id": correlation_id, "plan_summary": plan_summary, "routing": routing})
            if body.wait_seconds > 0:
                with suppress(TimeoutError):
                    await svc.runs.wait(run.id, timeout=body.wait_seconds)
            payload = _goal_view(run, correlation_id, plan_summary, routing)
            status_code = 202 if not payload["finished"] else 200
            logger.info("goal run %s -> %s [%s]", run.id, payload["status"], correlation_id)
            return JSONResponse(payload, status_code=status_code, headers={"X-Correlation-Id": correlation_id})
        except GoalError as exc:
            return _error_response(exc, correlation_id)

    @router.get("/api/v1/goals/{run_id}", summary="Status and result of a goal run")
    async def get_goal(  # pyright: ignore[reportUnusedFunction]
        run_id: str, who: Caller, response: Response, x_correlation_id: CorrelationHeader = None
    ) -> Any:
        remembered = recall(run_id)
        correlation_id = str(remembered.get("correlation_id") or (x_correlation_id or "").strip() or uuid.uuid4())
        response.headers["X-Correlation-Id"] = correlation_id
        try:
            try:
                run = services().runs.get(run_id)
            except run_error as exc:
                raise GoalError(404, "not_found", f"unknown run '{run_id}'") from exc
            if who.role != "admin" and run.principal.name != who.name:
                raise GoalError(404, "not_found", f"unknown run '{run_id}'")
            plan_summary = remembered.get("plan_summary") or _plan_summary(run.spec)
            payload = _goal_view(run, correlation_id, plan_summary, remembered.get("routing"))
            return JSONResponse(payload, headers={"X-Correlation-Id": correlation_id})
        except GoalError as exc:
            return _error_response(exc, correlation_id)

    return router