"""Workflow builder API: reusable templates, live validation, binding suggestions and drafts from a goal.

No `from __future__ import annotations` here: FastAPI must evaluate `Depends(...)` on closure-local functions.
"""

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from ueoa.api.app import Services, Settings, available_capabilities
from ueoa.contracts import CapabilityContract, NodeType, WorkflowSpec
from ueoa.gateway.service import Principal
from ueoa.orchestrator.bindings import auto_wire, suggest_bindings
from ueoa.orchestrator.validation import validate_spec
from ueoa.registry.seed import load_workflow
from ueoa.store import StoreError

KIND = "workflows"
PLATFORM_OWNER = "platform"

logger = logging.getLogger(__name__)


class TemplateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    workflow: dict[str, Any]
    inputs_example: dict[str, Any] | None = None


class WorkflowRequest(BaseModel):
    workflow: dict[str, Any]


class NodeRequest(WorkflowRequest):
    node_id: str = Field(min_length=1, max_length=100)


class DraftRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    inputs: dict[str, Any] = Field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _issue(code: str, message: str, node: str | None) -> dict[str, Any]:
    return {"code": code, "message": message, "node": node}


def _check(
    workflow: Mapping[str, Any], capabilities: Mapping[str, CapabilityContract]
) -> tuple[WorkflowSpec | None, list[dict[str, Any]]]:
    try:
        spec = WorkflowSpec.model_validate(workflow)
    except ValidationError as exc:
        return None, [
            _issue("schema", f"{'.'.join(str(p) for p in error['loc']) or 'workflow'}: {error['msg']}", None)
            for error in exc.errors(include_url=False)
        ]
    return spec, [_issue(i.code, i.message, i.node_id) for i in validate_spec(spec, capabilities)]


def _doc(
    spec: WorkflowSpec,
    issues: list[dict[str, Any]],
    *,
    name: str,
    description: str,
    inputs_example: dict[str, Any],
    owner: str,
    builtin: bool,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": name,
        "description": description,
        "workflow": spec.model_dump(mode="json", exclude_none=True),
        "inputs_example": inputs_example,
        "owner": owner,
        "builtin": builtin,
        "updated_at": updated_at,
        "valid": not issues,
        "issues": issues,
    }


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    workflow: dict[str, Any] = doc["workflow"]
    nodes: list[dict[str, Any]] = workflow.get("nodes", [])
    return {
        "id": doc["id"],
        "name": doc["name"],
        "description": doc["description"],
        "builtin": doc["builtin"],
        "owner": doc["owner"],
        "updated_at": doc["updated_at"],
        "valid": doc["valid"],
        "nodes": len(nodes),
        "capabilities": list(dict.fromkeys(n["capability"] for n in nodes if n.get("capability"))),
        "inputs": list(workflow.get("inputs", {})),
    }


def _load_builtins(
    catalog_dir: Path | None, capabilities: Mapping[str, CapabilityContract]
) -> dict[str, dict[str, Any]]:
    folder = catalog_dir / "workflows" if catalog_dir is not None else None
    if folder is None or not folder.is_dir():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.yaml")):
        if path.name.endswith(".design.yaml"):
            continue
        try:
            spec = load_workflow(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            logger.warning("skipping built-in workflow %s: %s", path.name, exc)
            continue
        _, issues = _check(spec.model_dump(mode="json", exclude_none=True), capabilities)
        found[spec.id] = _doc(
            spec,
            issues,
            name=spec.id,
            description=spec.goal,
            inputs_example={},
            owner=PLATFORM_OWNER,
            builtin=True,
            updated_at=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        )
    return found


async def on_startup(services: Services, settings: Settings) -> None:
    """Put the pack's built-in templates into the store (so runs by id work) and drop ones no longer shipped."""
    store = services.store
    builtins = _load_builtins(settings.catalog_dir, services.registry.capabilities)
    for doc in store.list(KIND):
        if doc.get("builtin") and doc.get("id") not in builtins:
            store.delete(KIND, doc["id"])
    for workflow_id, doc in builtins.items():
        try:
            current = store.get(KIND, workflow_id)
            if current is not None and not current.get("builtin"):
                logger.warning("built-in workflow '%s' replaces a saved template with the same id", workflow_id)
            store.put(KIND, workflow_id, doc)
        except StoreError as exc:
            logger.warning("skipping built-in workflow '%s': %s", workflow_id, exc)


def create_router(
    services: Callable[[], Services],
    principal: Callable[..., Principal],
    admin: Callable[..., Principal],
    settings: Settings,
) -> APIRouter:
    del admin, settings  # any role may use the builder; template edits are checked per owner
    router = APIRouter()
    Caller = Annotated[Principal, Depends(principal)]

    def stored(workflow_id: str) -> dict[str, Any] | None:
        try:
            return services().store.get(KIND, workflow_id)
        except StoreError:
            return None

    def fresh(doc: dict[str, Any]) -> dict[str, Any]:
        _, issues = _check(doc["workflow"], services().registry.capabilities)
        return {**doc, "valid": not issues, "issues": issues}

    def editable(workflow_id: str, who: Principal) -> dict[str, Any] | None:
        existing = stored(workflow_id)
        if existing is not None and existing.get("builtin"):
            raise HTTPException(409, "built-in templates are read-only; save under a new id")
        if existing is not None and existing.get("owner") != who.name and who.role != "admin":
            raise HTTPException(403, f"template '{workflow_id}' belongs to another user")
        return existing

    def parsed(workflow: dict[str, Any]) -> WorkflowSpec:
        try:
            return WorkflowSpec.model_validate(workflow)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc

    @router.get("/api/v1/workflows")
    def list_workflows(_: Caller) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        docs = sorted(services().store.list(KIND), key=lambda d: (not d.get("builtin"), d["id"]))
        return [_summary(fresh(doc)) for doc in docs]

    @router.get("/api/v1/workflows/{workflow_id}")
    def get_workflow(workflow_id: str, _: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        doc = stored(workflow_id)
        if doc is None:
            raise HTTPException(404, f"unknown workflow '{workflow_id}'")
        return fresh(doc)

    @router.put("/api/v1/workflows/{workflow_id}")
    def save_workflow(workflow_id: str, body: TemplateRequest, who: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        existing = editable(workflow_id, who)
        workflow = {**body.workflow}
        workflow.setdefault("id", workflow_id)
        if workflow["id"] != workflow_id:
            raise HTTPException(422, f"workflow.id '{workflow['id']}' must match the path id '{workflow_id}'")
        spec = parsed(workflow)
        _, issues = _check(spec.model_dump(mode="json", exclude_none=True), services().registry.capabilities)
        previous = existing or {}
        doc = _doc(
            spec,
            issues,
            name=body.name or previous.get("name") or spec.id,
            description=body.description if body.description is not None else previous.get("description", spec.goal),
            inputs_example=body.inputs_example
            if body.inputs_example is not None
            else previous.get("inputs_example", {}),
            owner=previous.get("owner", who.name),
            builtin=False,
            updated_at=_now(),
        )
        try:
            services().store.put(KIND, workflow_id, doc)
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc
        return doc

    @router.delete("/api/v1/workflows/{workflow_id}")
    def delete_workflow(workflow_id: str, who: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        if editable(workflow_id, who) is None:
            raise HTTPException(404, f"unknown workflow '{workflow_id}'")
        services().store.delete(KIND, workflow_id)
        return {"id": workflow_id, "deleted": True}

    @router.post("/api/v1/workflows/validate")
    def validate_workflow(body: WorkflowRequest, who: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        svc = services()
        spec, issues = _check(body.workflow, svc.registry.capabilities)
        warnings: list[dict[str, Any]] = []
        if spec is not None:
            available = set(available_capabilities(svc, who, include_writes=True))
            warnings = [
                _issue("not_available", f"'{node.capability}' has no approved implementation you may use", node.id)
                for node in spec.nodes
                if node.type is NodeType.CAPABILITY
                and node.capability in svc.registry.capabilities
                and node.capability not in available
            ]
        return {"ok": not issues, "issues": issues, "warnings": warnings}

    @router.post("/api/v1/workflows/suggest")
    def suggest(body: NodeRequest, _: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        spec = parsed(body.workflow)
        try:
            return suggest_bindings(spec, services().registry.capabilities, body.node_id)
        except KeyError as exc:
            raise HTTPException(422, f"unknown node '{body.node_id}'") from exc

    @router.post("/api/v1/workflows/autowire")
    def autowire(body: NodeRequest, _: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        spec = parsed(body.workflow)
        try:
            before = spec.node(body.node_id).inputs
            wired = auto_wire(spec, services().registry.capabilities, body.node_id)
        except KeyError as exc:
            raise HTTPException(422, f"unknown node '{body.node_id}'") from exc
        changed = [name for name in wired.node(body.node_id).inputs if name not in before]
        return {"workflow": wired.model_dump(mode="json", exclude_none=True), "changed": changed}

    @router.post("/api/v1/workflows/draft")
    async def draft(body: DraftRequest, who: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        svc = services()
        if svc.planner is None:
            raise HTTPException(503, "planner is not configured (set the UEOA_LLM_* settings)")
        available = available_capabilities(svc, who)
        result = (await svc.planner.plan(body.goal, body.inputs, svc.registry.capabilities, available)).view()
        return {key: result[key] for key in ("ok", "rationale", "issues", "warnings", "workflow")}

    @router.get("/api/v1/capabilities/{capability_id}/schema")
    def capability_schema(capability_id: str, _: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        registry = services().registry
        capability = registry.capabilities.get(capability_id)
        if capability is None:
            raise HTTPException(404, f"unknown capability '{capability_id}'")
        return {
            "id": capability.id,
            "description": capability.description,
            "input_schema": capability.input_schema,
            "output_schema": capability.output_schema,
            "selection_doc": capability.selection_doc.model_dump() if capability.selection_doc else None,
            "effects": str(capability.effects),
            "implementations": [
                {"key": impl.key, "status": str(impl.status)} for impl in registry.implementations_of(capability_id)
            ],
        }

    return router
