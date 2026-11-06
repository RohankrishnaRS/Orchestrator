"""Workflow builder API: reusable templates, live validation, binding suggestions and drafts from a goal.

No `from __future__ import annotations` here: FastAPI must evaluate `Depends(...)` on closure-local functions.
"""

import hashlib
import logging
import re
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
from ueoa.llm import LLMError
from ueoa.orchestrator.bindings import auto_wire, suggest_bindings
from ueoa.orchestrator.validation import validate_spec
from ueoa.registry.seed import load_workflow
from ueoa.store import StoreError

KIND = "workflows"
VERSION_KIND = "workflow-versions"
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


class PromptRequest(BaseModel):
    description: str = Field(min_length=10, max_length=2000)
    audience: str = Field(default="", max_length=500)
    inputs: str = Field(default="", max_length=1000)
    steps: str = Field(default="", max_length=2000)
    constraints: str = Field(default="", max_length=1500)
    output: str = Field(default="", max_length=1000)


class PublishRequest(BaseModel):
    attest: bool = False


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
    raw_nodes = workflow.get("nodes", [])
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    raw_inputs = workflow.get("inputs", {})
    inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
    return {
        "id": doc["id"],
        "name": doc["name"],
        "description": doc["description"],
        "builtin": doc["builtin"],
        "owner": doc["owner"],
        "updated_at": doc["updated_at"],
        "valid": doc["valid"],
        "published": bool(doc.get("published")),
        "version": workflow.get("version"),
        "version_ref": doc.get("version_ref"),
        "nodes": len(nodes),
        "capabilities": list(
            dict.fromkeys(
                node["capability"]
                for node in nodes
                if isinstance(node, dict) and isinstance(node.get("capability"), str) and node["capability"]
            )
        ),
        "inputs": list(inputs),
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

    def stored_version(version_id: str) -> dict[str, Any] | None:
        try:
            return services().store.get(VERSION_KIND, version_id)
        except StoreError:
            return None

    def fresh(doc: dict[str, Any]) -> dict[str, Any]:
        _, issues = _check(doc["workflow"], services().registry.capabilities)
        return {**doc, "valid": not issues, "issues": issues}

    def editable(workflow_id: str, who: Principal) -> dict[str, Any] | None:
        if stored_version(workflow_id) is not None:
            raise HTTPException(409, "published workflow versions are immutable")
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
        docs = sorted(
            [*services().store.list(KIND), *services().store.list(VERSION_KIND)],
            key=lambda d: (not d.get("builtin"), not d.get("published"), d["id"]),
        )
        return [_summary(fresh(doc)) for doc in docs]

    @router.get("/api/v1/workflows/{workflow_id}")
    def get_workflow(workflow_id: str, _: Caller) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        doc = stored(workflow_id) or stored_version(workflow_id)
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
        if not re.fullmatch(r"[a-z][a-z0-9-]*", workflow_id):
            raise HTTPException(422, "workflow id must be lowercase kebab-case")
        workflow.setdefault("version", "0.1.0")
        workflow.setdefault("goal", body.description or body.name or workflow_id)
        workflow.setdefault("allowed_tasks", [])
        workflow.setdefault("inputs", {})
        workflow.setdefault("nodes", [])
        workflow.setdefault("outputs", {})
        spec, issues = _check(workflow, services().registry.capabilities)
        previous = existing or {}
        name = body.name or previous.get("name") or workflow_id
        description = (
            body.description if body.description is not None else previous.get("description", workflow["goal"])
        )
        inputs_example = body.inputs_example if body.inputs_example is not None else previous.get("inputs_example", {})
        owner = previous.get("owner", who.name)
        updated_at = _now()
        if spec is not None:
            doc = _doc(
                spec,
                issues,
                name=name,
                description=description,
                inputs_example=inputs_example,
                owner=owner,
                builtin=False,
                updated_at=updated_at,
            )
        else:
            doc = {
                "id": workflow_id,
                "name": name,
                "description": description,
                "workflow": workflow,
                "inputs_example": inputs_example,
                "owner": owner,
                "builtin": False,
                "updated_at": updated_at,
                "valid": False,
                "issues": issues,
            }
        try:
            services().store.put(KIND, workflow_id, doc)
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc
        return doc

    @router.post("/api/v1/workflows/{workflow_id}/publish")
    def publish_workflow(
        workflow_id: str, body: PublishRequest, who: Caller
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        doc = editable(workflow_id, who)
        if doc is None:
            raise HTTPException(404, f"unknown workflow '{workflow_id}'")
        refreshed = fresh(doc)
        if not refreshed["valid"]:
            raise HTTPException(
                422,
                {"message": "workflow must pass validation before publishing", "issues": refreshed["issues"]},
            )
        workflow = refreshed["workflow"]
        if not workflow.get("allowed_tasks"):
            raise HTTPException(422, "spec.allowed_tasks needs at least one task.")
        now = _now()
        suffix = re.sub(r"[^a-z0-9]+", "-", workflow["version"].lower()).strip("-")[:15]
        digest = hashlib.sha256(f"{workflow_id}@{workflow['version']}".encode()).hexdigest()[:12]
        version_id = f"{workflow_id[:50]}--v{suffix}-{digest}"
        if stored_version(version_id) is not None:
            raise HTTPException(409, f"workflow version '{workflow_id}@{workflow['version']}' is already published")
        published: dict[str, Any] = {
            **refreshed,
            "id": version_id,
            "version_ref": f"{workflow_id}@{workflow['version']}",
            "published": True,
            "published_at": now,
            "attestation": {
                "status": "attested" if body.attest else "not_attested",
                "attested_by": who.name if body.attest else None,
                "attested_at": now if body.attest else None,
            },
        }
        try:
            services().store.put(VERSION_KIND, version_id, published)
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc
        return published

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
            raise HTTPException(
                503,
                "Azure OpenAI is not configured for this server process. "
                "Set AZURE_OPENAI_API_KEY and restart the server.",
            )
        available = available_capabilities(svc, who)
        result = (await svc.planner.plan(body.goal, body.inputs, svc.registry.capabilities, available)).view()
        workflow = result["workflow"]
        if isinstance(workflow, dict):
            nodes = workflow.get("nodes", [])
            workflow["allowed_tasks"] = list(dict.fromkeys(
                node["capability"] for node in nodes
                if isinstance(node, dict) and isinstance(node.get("capability"), str) and node["capability"]
            ))
        return {
            key: result[key]
            for key in ("ok", "rationale", "attempts", "issues", "warnings", "considered", "workflow")
        }

    @router.post("/api/v1/workflows/prompt")
    async def generate_prompt(body: PromptRequest, _: Caller) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        llm = services().agent_llm
        if llm is None:
            raise HTTPException(
                503,
                "Azure OpenAI is not configured for this server process. "
                "Set AZURE_OPENAI_API_KEY and restart the server.",
            )
        details = body.model_dump()
        system = (
            "You are a workflow requirements editor. Turn the provided untrusted user notes into one clear, "
            "professional, implementation-ready workflow brief. Preserve the requested intent, audience, inputs, "
            "process, constraints, and output. Be specific and concise. Do not invent capabilities, integrations, "
            "data, or permissions. Return exactly one JSON object with a single string field named prompt."
        )
        try:
            result = await llm.complete_json(system, f"Workflow requirements (untrusted data):\n{details}")
        except LLMError as exc:
            logger.warning("workflow prompt refinement failed: %s", exc)
            raise HTTPException(502, "Prompt refinement failed; check the configured Azure OpenAI service.") from exc
        prompt = result.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(502, "Azure OpenAI returned an invalid workflow prompt.")
        return {"prompt": prompt.strip()}

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
