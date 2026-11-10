"""UEOA API: registry, gateway, planner and run endpoints plus the demo page.

No `from __future__ import annotations` here: FastAPI must evaluate `Depends(...)` on closure-local functions.
"""

import importlib
import logging
import os
import re
import ssl
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import truststore
import yaml
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ueoa.api.mcp import create_mcp_router, origin_allowed
from ueoa.contracts import CapabilityContract, Effect, InputSpec, NodeType, WorkflowSpec
from ueoa.gateway.auth import EntraTokenProvider, StaticTokenProvider, TokenProvider
from ueoa.gateway.invokers import Invoker
from ueoa.gateway.service import (
    Entitlements,
    Gateway,
    GatewayDeniedError,
    GatewayFailedError,
    InvocationRecord,
    Principal,
)
from ueoa.llm import AzureOpenAIJsonLLM, JsonLLM, LLMError
from ueoa.orchestrator.expressions import ExpressionError, parse, references
from ueoa.orchestrator.graph import node_expressions
from ueoa.orchestrator.planner import Planner
from ueoa.orchestrator.runner import PendingApproval, Run, RunController, RunError
from ueoa.persistence import Stores, build_stores
from ueoa.persistence.writebehind import flush_writes, schedule_write
from ueoa.registry.models import DEFAULT_ENTRA_SCOPE, Implementation, ImplementationStatus
from ueoa.registry.seed import load_capabilities
from ueoa.registry.service import Registry, RegistryError
from ueoa.store import JsonStore

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_ENTITLEMENTS = {"admin": ["*"], "user": ["*"]}
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
NODE_TIMEOUT_HEADROOM_S = 15

logger = logging.getLogger(__name__)


def _load_environment(env_file: Path | None = None) -> None:
    load_dotenv(env_file or ROOT / ".env", override=False)


def _env_list(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in (os.environ.get(name) or default).split(",") if item.strip())


@dataclass
class Settings:
    # Optional seed pack (capabilities/, implementations/, entitlements.yaml). The platform itself ships none.
    catalog_dir: Path | None = field(
        default_factory=lambda: Path(os.environ["UEOA_CATALOG_DIR"]) if os.environ.get("UEOA_CATALOG_DIR") else None
    )
    allowed_hosts: tuple[str, ...] = field(
        default_factory=lambda: _env_list("UEOA_ALLOWED_HOSTS", "localhost,127.0.0.1")
    )
    local_tool_prefixes: tuple[str, ...] = field(default_factory=lambda: _env_list("UEOA_LOCAL_TOOL_PREFIXES"))
    default_role: str = field(default_factory=lambda: os.environ.get("UEOA_DEFAULT_ROLE", "user"))
    planner_enabled: bool = field(default_factory=lambda: os.environ.get("UEOA_PLANNER", "on").lower() != "off")
    # DEV ONLY: fixed bearer for `entra` agents on loopback hosts (local mocks); other hosts use DefaultAzureCredential.
    entra_static_token: str = field(default_factory=lambda: os.environ.get("UEOA_ENTRA_STATIC_TOKEN", ""))
    entra_scopes: tuple[str, ...] = field(default_factory=lambda: _env_list("UEOA_ENTRA_SCOPES", DEFAULT_ENTRA_SCOPE))
    trusted_hosts: tuple[str, ...] = field(
        default_factory=lambda: _env_list("UEOA_TRUSTED_HOSTS", "localhost,127.0.0.1,testserver,*.localhost")
    )
    # Empty = a localhost / 127.0.0.1 origin on the port the request came in on.
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: _env_list("UEOA_ALLOWED_ORIGINS"))
    max_body_bytes: int = field(default_factory=lambda: int(os.environ.get("UEOA_MAX_BODY_BYTES") or 5_000_000))
    # Lets the requester approve their own writes (demo only); four-eyes is the default.
    allow_self_approval: bool = field(
        default_factory=lambda: os.environ.get("UEOA_ALLOW_SELF_APPROVAL", "").lower() == "true"
    )
    directline_poll_s: float = 1.0
    # Where hosted agents, workflow templates etc. are saved; empty = memory only.
    data_dir: Path | None = field(
        default_factory=lambda: Path(os.environ["UEOA_DATA_DIR"]) if os.environ.get("UEOA_DATA_DIR") else None
    )
    # How agents hosted by this platform are addressed in their cards (must be reachable by the gateway).
    public_base_url: str = field(
        default_factory=lambda: os.environ.get("UEOA_PUBLIC_BASE_URL", "").rstrip("/")
    )


@dataclass
class Services:
    registry: Registry
    gateway: Gateway
    entitlements: Entitlements
    runs: RunController
    planner: Planner | None = None
    token_provider: TokenProvider | None = None
    # capability id -> minimum node timeout (card timeout + headroom), None when unresolvable
    node_timeout: Callable[[str], float | None] | None = None
    runner_applies_node_timeout: bool = False
    store: JsonStore = field(default_factory=JsonStore)
    stores: Stores | None = None
    # model used by agents the platform hosts itself (agent studio); None = hosting disabled
    agent_llm: JsonLLM | None = None


async def build_services(settings: Settings, http: httpx.AsyncClient, stores: Stores | None = None) -> Services:
    catalog = settings.catalog_dir
    capabilities = load_capabilities(catalog / "capabilities") if catalog else {}
    catalog_capability_ids = set(capabilities)
    if stores is not None and stores.capabilities is not None:
        for capability in capabilities.values():
            await stores.capabilities.upsert(capability)
        stored_capabilities = await stores.capabilities.list()
        for capability in stored_capabilities:
            current = capabilities.get(capability.id)
            if current is None or (
                capability.id not in catalog_capability_ids
                and tuple(map(int, capability.version.split(".")))
                > tuple(map(int, current.version.split(".")))
            ):
                capabilities[capability.id] = capability
    token_provider = EntraTokenProvider()
    loopback_token_provider = None
    if settings.entra_static_token:
        logger.warning("UEOA_ENTRA_STATIC_TOKEN is set: a fixed DEV token is sent to loopback Entra agents only")
        loopback_token_provider = StaticTokenProvider(settings.entra_static_token)
    invoker = Invoker(
        http,
        poll_interval_s=settings.directline_poll_s,
        local_prefixes=settings.local_tool_prefixes,
        token_provider=token_provider,
        loopback_token_provider=loopback_token_provider,
        entra_scopes=settings.entra_scopes,
    )
    registry = Registry(capabilities, invoker, http, settings.allowed_hosts, entra_scopes=settings.entra_scopes)
    if stores is not None and stores.implementations is not None:
        registry.restore(await stores.implementations.list())
    if catalog:
        for path in sorted((catalog / "implementations").glob("*.yaml")):
            for raw in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
                impl = await registry.register_seed(raw)
                if stores is not None and stores.implementations is not None:
                    await stores.implementations.upsert(impl, seed=True)
    entitlements_file = catalog / "entitlements.yaml" if catalog else None
    roles = (
        yaml.safe_load(entitlements_file.read_text(encoding="utf-8"))
        if entitlements_file and entitlements_file.is_file()
        else DEFAULT_ENTITLEMENTS
    )
    entitlements = Entitlements(roles)
    gateway = Gateway(
        registry,
        invoker,
        entitlements,
        allow_self_approval=settings.allow_self_approval,
        audit_writer=stores.audit if stores is not None else None,
    )

    def node_timeout(capability_id: str) -> float | None:
        try:
            return registry.resolve(capability_id).card.ueoa.timeout_s + NODE_TIMEOUT_HEADROOM_S
        except RegistryError:
            return None

    try:
        runs = RunController(
            gateway,
            lambda: registry.capabilities,
            node_timeout=node_timeout,
            persist_run=stores.runs.persist if stores is not None and stores.runs is not None else None,
            persist_approval=stores.approvals.decide if stores is not None and stores.approvals is not None else None,
        )
        applies = True
    except TypeError:  # runner without the node_timeout hook: the API raises timeouts itself
        runs = RunController(
            gateway,
            lambda: registry.capabilities,
            persist_run=stores.runs.persist if stores is not None and stores.runs is not None else None,
            persist_approval=stores.approvals.decide if stores is not None and stores.approvals is not None else None,
        )
        applies = False
    return Services(
        registry,
        gateway,
        entitlements,
        runs,
        token_provider=token_provider,
        node_timeout=node_timeout,
        runner_applies_node_timeout=applies,
        store=JsonStore(settings.data_dir),
        stores=stores,
    )


class BodyLimitMiddleware:
    """413 for bodies over `max_bytes`, by Content-Length and while streaming."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = Headers(scope=scope).get("content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            status = 413 if length.isdigit() else 400
            await JSONResponse({"detail": "request body too large or invalid"}, status)(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, limited_receive, send)


class OriginGuardMiddleware:
    """Browsers send Origin on cross-site writes; reject state-changing requests from any origin not allowed."""

    def __init__(self, app: ASGIApp, allowed_origins: tuple[str, ...]) -> None:
        self.app = app
        self.allowed_origins = allowed_origins

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] not in SAFE_METHODS
            and not origin_allowed(HTTPConnection(scope), self.allowed_origins)
        ):
            await JSONResponse({"detail": "origin not allowed"}, 403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def available_capabilities(services: Services, who: Principal, *, include_writes: bool = False) -> list[str]:
    """Capabilities the caller may run now: entitled, with an approved implementation, read-only unless asked."""
    gateway_available = getattr(services.gateway, "available", None)
    if gateway_available is not None:
        return list(gateway_available(who, include_writes=include_writes))
    found: list[str] = []
    for capability_id, capability in services.registry.capabilities.items():
        if not services.entitlements.allows(who.role, capability_id):
            continue
        if capability.effects is not Effect.READ and not include_writes:
            continue
        try:
            services.registry.resolve(capability_id)
        except RegistryError:
            continue
        found.append(capability_id)
    return sorted(found)


class RegisterRequest(BaseModel):
    url: str
    overlay: dict[str, Any] | None = None  # UEOA extension params for agents whose card lacks them


class ApproveRequest(BaseModel):
    priority: int | None = None


class InvokeRequest(BaseModel):
    capability: str
    input: dict[str, Any]


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    inputs: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    goal: str | None = Field(default=None, min_length=1, max_length=2000)
    workflow: dict[str, Any] | None = None
    workflow_id: str | None = Field(default=None, min_length=1, max_length=81)
    inputs: dict[str, Any] = Field(default_factory=dict)
    wait_s: float = Field(default=0, ge=0, le=300)


_GOAL_INPUT_NAMES = (
    "goal",
    "prompt",
    "user_prompt",
    "query",
    "request",
    "user_request",
    "task",
    "task_description",
    "code_request",
    "instruction",
    "user_input",
    "input_text",
    "text",
    "message",
    "description",
    "objective",
    "requirements",
    "requirement",
    "question",
    "content",
    "input",
)
_GOAL_HINTS = frozenset(
    {
        "goal",
        "prompt",
        "query",
        "request",
        "task",
        "instruction",
        "input",
        "text",
        "message",
        "description",
        "objective",
        "requirement",
        "question",
        "content",
        "code",
    }
)


def _is_string_field(field: Any) -> bool:
    field_type = field.type if isinstance(field, InputSpec) else field.get("type") if isinstance(field, dict) else None
    return field_type == "string" or (isinstance(field_type, list) and "string" in field_type)


def _select_goal_field(fields: dict[str, Any], required: set[str] | None = None) -> str | None:
    string_fields = {name: field for name, field in fields.items() if _is_string_field(field)}
    for name in _GOAL_INPUT_NAMES:
        if name in string_fields:
            return name
    if len(string_fields) == 1:
        return next(iter(string_fields))
    if required:
        required_strings = [name for name in string_fields if name in required]
        if len(required_strings) == 1:
            return required_strings[0]

    def score(name: str, field: Any) -> int:
        description = field.description if isinstance(field, InputSpec) else field.get("description", "")
        words = set(re.findall(r"[a-z0-9]+", f"{name} {description}".lower()))
        return len(words & _GOAL_HINTS)

    scored = [(score(name, field), name) for name, field in string_fields.items()]
    if not scored:
        return None
    highest = max(value for value, _ in scored)
    best = [name for value, name in scored if value == highest and value > 0]
    return best[0] if len(best) == 1 else None


def _workflow_goal_inputs(spec: WorkflowSpec, goal: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Bind the chat goal to a declared workflow input without changing saved workflow definitions."""
    target = next((name for name in _GOAL_INPUT_NAMES if name in spec.inputs), None)
    if target is None:
        target = _select_goal_field(spec.inputs)
    if target is None:
        object_inputs = [name for name, declaration in spec.inputs.items() if declaration.type == "object"]
        if len(spec.inputs) == 1 or (
            len(object_inputs) == 1 and not any(_is_string_field(declaration) for declaration in spec.inputs.values())
        ):
            target = object_inputs[0] if object_inputs else None
    if target is None:
        available = ", ".join(spec.inputs) or "(none)"
        raise HTTPException(
            422,
            "Could not identify a text input for the chat goal in this workflow. "
            f"Declared inputs: {available}. Rename the task field to a goal/prompt/request field "
            "or provide an unambiguous text field in Workflow inputs (JSON).",
        )

    declaration: InputSpec = spec.inputs[target]
    if declaration.type != "object":
        if declaration.type != "string":
            raise HTTPException(422, f"Workflow input '{target}' must be a string or an object to receive the goal.")
        return {**inputs, target: goal}

    schema = declaration.model_extra or {}
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    property_name = next((name for name in _GOAL_INPUT_NAMES if name in properties), None)
    if property_name is None:
        required = schema.get("required", [])
        property_name = _select_goal_field(properties, set(required) if isinstance(required, list) else None)
    if property_name is None and schema.get("additionalProperties", True) is not False:
        property_name = "goal"
    if property_name is None:
        available = ", ".join(properties) or "(none)"
        raise HTTPException(
            422,
            f"Workflow input '{target}' has no unambiguous string field for the chat goal. "
            f"Declared fields: {available}. Add a goal/prompt/request field.",
        )
    existing = inputs.get(target, {})
    if not isinstance(existing, dict):
        raise HTTPException(422, f"Workflow input '{target}' must be a JSON object to receive the goal.")
    return {**inputs, target: {**existing, property_name: goal}}


def _referenced_workflow_inputs(spec: WorkflowSpec) -> set[str]:
    found: set[str] = set()
    expressions = [expression.source for node in spec.nodes for expression in node_expressions(node)]
    expressions.extend(spec.outputs.values())
    for expression in expressions:
        try:
            found.update(
                str(use.ref.path[0])
                for use in references(parse(expression))
                if use.ref.root == "inputs" and use.ref.path
            )
        except ExpressionError:
            continue
    return found


def _prepare_saved_workflow(
    spec: WorkflowSpec,
    goal: str,
    inputs: dict[str, Any],
    capabilities: Mapping[str, CapabilityContract],
) -> tuple[WorkflowSpec, dict[str, Any]]:
    """Bind the chat goal and omit unset optional tool arguments from this run only."""
    spec = spec.model_copy(deep=True)
    run_inputs = _workflow_goal_inputs(spec, goal, inputs)
    missing = {
        name
        for name, declaration in spec.inputs.items()
        if name not in run_inputs and "default" not in declaration.model_fields_set
    }

    for node in spec.nodes:
        if node.type is not NodeType.CAPABILITY or not node.capability:
            continue
        capability = capabilities.get(node.capability)
        if capability is None:
            continue
        schema = capability.input_schema
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        if not isinstance(properties, dict):
            continue
        for input_name, binding in list(node.inputs.items()):
            if not isinstance(binding, str):
                continue
            try:
                used_inputs = {
                    str(use.ref.path[0])
                    for use in references(parse(binding))
                    if use.ref.root == "inputs" and use.ref.path
                }
            except ExpressionError:
                continue
            if used_inputs & missing and input_name in properties and input_name not in required:
                node.inputs.pop(input_name)

    still_required = missing & _referenced_workflow_inputs(spec)
    if still_required:
        raise HTTPException(422, f"workflow inputs required for this task: {', '.join(sorted(still_required))}")
    referenced = _referenced_workflow_inputs(spec)
    spec.inputs = {
        name: declaration
        for name, declaration in spec.inputs.items()
        if name not in missing or name in referenced
    }
    return spec, run_inputs


class HumanInputRequest(BaseModel):
    value: dict[str, Any]


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    payload_hash: str = Field(min_length=1, max_length=128)
    item: int | None = Field(default=None, ge=0)


def _approval_view(run: Run, pending: PendingApproval) -> dict[str, Any]:
    return {"run_id": run.id, "goal": run.spec.goal, **asdict(pending)}


def _impl_view(impl: Implementation) -> dict[str, Any]:
    card = impl.card.model_dump(mode="json", by_alias=True, exclude_none=True)
    ext = impl.card.ueoa
    return {
        "key": impl.key,
        "name": impl.card.name,
        "kind": ext.kind,
        "implementation": ext.implementation,
        "binding": ext.binding,
        "owner": ext.owner,
        "implements": ext.implements,
        "status": impl.status,
        "priority": impl.priority,
        "source": impl.source,
        "card_sha256": impl.card_sha256,
        "conformant": impl.conformant,
        "checks": [check.model_dump() for check in impl.checks],
        "skills": [skill.model_dump() for skill in impl.card.skills],
        "card": card,
    }


def _record_view(record: InvocationRecord) -> dict[str, Any]:
    return asdict(record)


def _card_view(impl: Implementation) -> dict[str, Any]:
    return {
        "key": impl.key,
        "name": impl.card.name,
        "description": impl.card.description,
        "url": impl.source,
        "binding": impl.card.ueoa.binding,
        "implements": impl.card.ueoa.implements,
        "skills": [{"id": s.id, "name": s.name, "description": s.description} for s in impl.card.skills],
        "status": impl.status,
        "priority": impl.priority,
        "card": impl.card.model_dump(mode="json", by_alias=True, exclude_none=True),
    }


NOT_RELEVANT = "not relevant to the goal (planner)"
NOT_USED = "not used by the workflow"
NOT_PERMITTED = "not permitted for role"
NOT_OFFERED = "write capability (not offered to the planner)"


def routing_view(
    services: Services, who: Principal, spec: WorkflowSpec | None, available: list[str], considered: list[str]
) -> dict[str, Any]:
    """Which agent each node routes to now, and which approved agents were left out and why (no model call)."""
    registry = services.registry

    def agent(impl: Implementation | None) -> dict[str, Any]:
        if impl is None:
            return {"implementation": None, "agent_name": None, "binding": None}
        return {"implementation": impl.key, "agent_name": impl.card.name, "binding": impl.card.ueoa.binding}

    def resolved(capability_id: str) -> Implementation | None:
        try:
            return registry.resolve(capability_id)
        except RegistryError:
            return None

    nodes = [n for n in spec.nodes if n.type is NodeType.CAPABILITY and n.capability] if spec else []
    selected = [
        {
            "node": n.id,
            "capability": n.capability,
            **agent(resolved(n.capability or "")),
            "fan_out": n.for_each is not None,
        }
        for n in nodes
    ]
    used = {n.capability for n in nodes}
    unused_reason = NOT_RELEVANT if considered else NOT_USED
    not_selected = [
        {"capability": c, **agent(resolved(c)), "reason": unused_reason} for c in available if c not in used
    ]
    offered = set(available) | used
    for impl in registry.implementations():
        if impl.status is not ImplementationStatus.APPROVED:
            continue
        for tag in impl.card.ueoa.implements:
            capability_id = tag.rsplit("@", 1)[0]
            if capability_id not in registry.capabilities or capability_id in offered:
                continue
            reason = NOT_PERMITTED if not services.entitlements.allows(who.role, capability_id) else NOT_OFFERED
            not_selected.append({"capability": capability_id, **agent(impl), "reason": reason})
    return {"selected": selected, "not_selected": not_selected, "considered": list(considered)}


def _build_planner(settings: Settings, llm: JsonLLM | None, client: httpx.AsyncClient) -> Planner | None:
    if llm is not None:
        return Planner(llm)
    if not settings.planner_enabled:
        return None
    try:
        if azure_llm := AzureOpenAIJsonLLM.from_env(client):
            return Planner(azure_llm)
    except LLMError:
        logger.exception("Azure OpenAI configuration is invalid")
        return None
    logger.warning("Azure OpenAI API key is not configured; workflow planning is unavailable")
    return None


def _build_agent_llm(llm: JsonLLM | None, client: httpx.AsyncClient) -> JsonLLM | None:
    if llm is not None:
        return llm
    try:
        if azure_llm := AzureOpenAIJsonLLM.from_env(client):
            return azure_llm
    except LLMError:
        logger.exception("Azure OpenAI configuration is invalid")
        return None
    logger.warning("Azure OpenAI API key is not configured; AI agent generation is unavailable")
    return None


def _optional_module(name: str) -> Any:
    """Feature routers are optional modules; the platform still starts without them."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None


def create_app(
    settings: Settings | None = None,
    http: httpx.AsyncClient | None = None,
    planner_llm: JsonLLM | None = None,
    agent_llm: JsonLLM | None = None,
) -> FastAPI:
    if settings is None:
        _load_environment()
        settings = Settings()
    agents_api = _optional_module("ueoa.api.agents")
    workflows_api = _optional_module("ueoa.api.workflows")
    skills_api = _optional_module("ueoa.api.skills")
    mcp_catalog_api = _optional_module("ueoa.api.mcp_catalog")
    goals_api = _optional_module("ueoa.api.goals")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Calls are bounded per invocation by asyncio timeouts; the client limit only guards against hangs.
        client = http or httpx.AsyncClient(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=httpx.Timeout(600, connect=10)
        )
        stores = build_stores()
        app.state.stores = stores
        try:
            if stores.backend == "postgres" and not await stores.healthy():
                raise RuntimeError(
                    "UEOA_STORE_URL is set, but PostgreSQL is unreachable. Check the URL and start the database."
                )
            app.state.services = await build_services(settings, client, stores)
            app.state.services.planner = _build_planner(settings, planner_llm, client)
            app.state.services.agent_llm = _build_agent_llm(agent_llm or planner_llm, client)
            if stores.workflows is not None:
                for doc in await stores.workflows.list():
                    restored = dict(doc)
                    kind = str(restored.pop("_kind", "workflows"))
                    app.state.services.store.put(kind, str(restored["id"]), restored)
            if stores.runs is not None:
                app.state.services.runs.restore(await stores.runs.list_recent(limit=200))
            for module in (agents_api, workflows_api, skills_api, mcp_catalog_api, goals_api):
                startup = getattr(module, "on_startup", None)
                if startup is not None:
                    await startup(app.state.services, settings)
            yield
        finally:
            svc = getattr(app.state, "services", None)
            if svc is not None:
                close_runs = getattr(svc.runs, "close", None)
                if close_runs is not None:
                    await close_runs()
                if isinstance(svc.token_provider, EntraTokenProvider):
                    await svc.token_provider.close()
            await flush_writes()
            await stores.close()
            if http is None:
                await client.aclose()

    app = FastAPI(title="UEOA - registry and gateway", version="0.1.0", lifespan=lifespan)
    # last added runs first: host check, then origin check, then the body limit
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(OriginGuardMiddleware, allowed_origins=settings.allowed_origins)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.trusted_hosts))

    def services() -> Services:
        return app.state.services  # type: ignore[no-any-return]

    def persistence_stores() -> Stores:
        return app.state.stores  # type: ignore[no-any-return]

    def principal(x_demo_role: Annotated[str | None, Header()] = None) -> Principal:
        role = (x_demo_role or settings.default_role).lower()
        if role not in services().entitlements.roles():
            raise HTTPException(400, f"unknown demo role '{role}'")
        return Principal(name=f"demo-{role}", role=role)

    def admin(who: Annotated[Principal, Depends(principal)]) -> Principal:
        if who.role != "admin":
            raise HTTPException(403, "only the Platform admin role can register or approve implementations")
        return who

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC / "index.html")

    @app.get("/create/workflow", include_in_schema=False)
    def workflow_studio() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC / "index.html")

    @app.get("/create/saved-workflows", include_in_schema=False)
    def saved_workflows() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC / "index.html")

    @app.get("/create/prompt", include_in_schema=False)
    def prompt_builder() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC / "index.html")

    @app.get("/api/v1/roles")
    def roles() -> list[str]:  # pyright: ignore[reportUnusedFunction]
        return services().entitlements.roles()

    @app.get("/api/v1/capabilities")
    def capabilities() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        registry = services().registry
        return [
            {
                **capability.model_dump(mode="json"),
                "implementations": [impl.key for impl in registry.implementations_of(capability.id)],
            }
            for capability in registry.capabilities.values()
        ]

    @app.post("/api/v1/capabilities", status_code=201)
    async def register_capability(  # pyright: ignore[reportUnusedFunction]
        contract: Annotated[dict[str, Any], Body()], _: Annotated[Principal, Depends(admin)]
    ) -> dict[str, Any]:
        try:
            capability = services().registry.register_capability(contract)
            stores = services().stores
            if stores is not None and stores.capabilities is not None:
                schedule_write(stores.capabilities.upsert(capability), f"capability {capability.id}")
            return capability.model_dump(mode="json")
        except RegistryError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/implementations")
    def implementations() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return [_impl_view(impl) for impl in services().registry.implementations()]

    @app.get("/api/v1/cards")
    def cards() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return [_card_view(impl) for impl in services().registry.implementations()]

    @app.get("/api/v1/implementations/{key}")
    def implementation(key: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        try:
            return _impl_view(services().registry.get(key))
        except RegistryError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/v1/implementations", status_code=201)
    async def register(  # pyright: ignore[reportUnusedFunction]
        body: RegisterRequest, _: Annotated[Principal, Depends(admin)]
    ) -> dict[str, Any]:
        try:
            impl = await services().registry.register_from_url(body.url, body.overlay)
            if (stores := services().stores) is not None and stores.implementations is not None:
                schedule_write(stores.implementations.upsert(impl), f"implementation {impl.key}")
            return _impl_view(impl)
        except RegistryError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/implementations/cards", status_code=201)
    async def submit_card(  # pyright: ignore[reportUnusedFunction]
        card: Annotated[dict[str, Any], Body()], _: Annotated[Principal, Depends(admin)]
    ) -> dict[str, Any]:
        try:
            impl = await services().registry.register_card(card)
            if (stores := services().stores) is not None and stores.implementations is not None:
                schedule_write(stores.implementations.upsert(impl), f"implementation {impl.key}")
            return _impl_view(impl)
        except RegistryError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/implementations/{key}/approve")
    async def approve(  # pyright: ignore[reportUnusedFunction]
        key: str, body: ApproveRequest, _: Annotated[Principal, Depends(admin)]
    ) -> dict[str, Any]:
        try:
            previous = services().registry.get(key).model_copy(deep=True)
        except RegistryError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            impl = services().registry.approve(key, body.priority)
            stores = services().stores
            if stores is not None and stores.implementations is not None:
                try:
                    await stores.implementations.upsert(impl)
                except Exception as exc:
                    impl.status = previous.status
                    impl.priority = previous.priority
                    logger.error("failed to persist implementation approval for %s (%s)", key, type(exc).__name__)
                    raise HTTPException(503, "implementation approval could not be persisted") from exc
            return _impl_view(impl)
        except RegistryError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/invoke")
    async def invoke(  # pyright: ignore[reportUnusedFunction]
        body: InvokeRequest, who: Annotated[Principal, Depends(principal)]
    ) -> dict[str, Any]:
        try:
            return _record_view(await services().gateway.invoke(who, body.capability, body.input))
        except GatewayDeniedError as exc:
            raise HTTPException(403, _record_view(exc.record)) from exc
        except GatewayFailedError as exc:
            raise HTTPException(502, _record_view(exc.record)) from exc

    @app.get("/api/v1/audit")
    async def audit(who: Annotated[Principal, Depends(principal)]) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        stores = services().stores
        if stores is not None and stores.audit is not None:
            records = await stores.audit.recent(limit=2000)
            return [
                {
                    "invocation_id": record["invocation_id"],
                    "at": record["at"],
                    "principal": {"name": record["principal"], "role": record["role"]},
                    "capability": record["capability"],
                    "status": record["status"],
                    "implementation": record["implementation"],
                    "duration_ms": record["duration_ms"],
                    "steps": record["checks"],
                    "output": record["output"],
                    "error": record["reason"],
                }
                for record in records
                if who.role == "admin" or record["principal"] == who.name
            ]
        return [
            _record_view(record)
            for record in reversed(services().gateway.audit)
            if who.role == "admin" or record.principal == who
        ]

    @app.get("/api/v1/health")
    async def health(
        stores: Annotated[Stores, Depends(persistence_stores)],
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        svc = services()
        impls = svc.registry.implementations()
        return {
            "status": "ok",
            "planner": svc.planner is not None,
            "capabilities": len(svc.registry.capabilities),
            "implementations": len(impls),
            "approved": sum(impl.status is ImplementationStatus.APPROVED for impl in impls),
            "default_role": settings.default_role,
            "store": {"backend": stores.backend, "ok": await stores.healthy()},
        }

    @app.get("/api/v1/overview")
    def overview(who: Annotated[Principal, Depends(principal)]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        svc = services()
        implementations = svc.registry.implementations()
        mcp_servers = {
            impl.card.ueoa.mcp.url
            for impl in implementations
            if impl.card.ueoa.mcp is not None and impl.status is not ImplementationStatus.REVOKED
        }
        mcp_servers.update(server["url"] for server in svc.store.list("mcp-servers"))
        hosted = svc.store.list("agents")
        hosted_keys = {
            f"{doc.get('definition', {}).get('id')}@{doc.get('definition', {}).get('version')}" for doc in hosted
        }
        approved_keys = {impl.key for impl in implementations if impl.status is ImplementationStatus.APPROVED}
        external_agents = [
            impl
            for impl in implementations
            if impl.card.ueoa.kind == "agent" and impl.key not in hosted_keys
        ]
        agents = len(hosted) + len(external_agents)
        active_agents = sum(impl.status is ImplementationStatus.APPROVED for impl in external_agents)
        active_agents += sum(key in approved_keys for key in hosted_keys)
        runs = [
            run
            for run in svc.runs.list()
            if who.role == "admin" or run.principal.name == who.name
        ]
        workflows = svc.store.list("workflows")
        skills = svc.store.list("skills")
        audit = [
            _record_view(record)
            for record in reversed(svc.gateway.audit)
            if who.role == "admin" or record.principal == who
        ]
        activities = [
            {
                "type": "run",
                "id": run.id,
                "label": run.spec.goal,
                "status": str(run.status),
                "at": run.finished_at or run.created_at,
            }
            for run in runs
        ] + [
            {
                "type": "invocation",
                "id": record["invocation_id"],
                "label": record["capability"],
                "status": record["status"],
                "at": record["at"],
            }
            for record in audit
        ]
        activities.sort(key=lambda activity: activity["at"], reverse=True)
        successful = sum(str(run.status) == "completed" for run in runs)
        failed = sum(str(run.status) in {"failed", "cancelled", "completed_rejected"} for run in runs)
        return {
            "total_agents": agents,
            "active_agents": active_agents,
            "total_workflows": len(workflows),
            "total_skills": len(skills),
            "registered_mcp_servers": len(mcp_servers),
            "total_runs": len(runs),
            "successful_runs": successful,
            "failed_runs": failed,
            "recent_activities": activities[:10],
        }

    def require_planner() -> Planner:
        planner = services().planner
        if planner is None:
            raise HTTPException(
                503,
                "Azure OpenAI is not configured for this server process. "
                "Set AZURE_OPENAI_API_KEY and restart the server.",
            )
        return planner

    @app.post("/api/v1/plans")
    async def plan(  # pyright: ignore[reportUnusedFunction]
        body: PlanRequest, who: Annotated[Principal, Depends(principal)]
    ) -> dict[str, Any]:
        planner = require_planner()
        svc = services()
        available = available_capabilities(svc, who)
        result = await planner.plan(body.goal, body.inputs, svc.registry.capabilities, available)
        routing = routing_view(svc, who, result.spec, available, result.considered)
        return {**result.view(), "available": available, "routing": routing}

    def owned_run(run_id: str, who: Principal) -> Run:
        try:
            run = services().runs.get(run_id)
        except RunError as exc:
            raise HTTPException(404, str(exc)) from exc
        if who.role != "admin" and run.principal.name != who.name:
            raise HTTPException(404, f"unknown run '{run_id}'")
        return run

    @app.post("/api/v1/runs", status_code=201)
    async def start_run(  # pyright: ignore[reportUnusedFunction]
        body: RunRequest, who: Annotated[Principal, Depends(principal)]
    ) -> dict[str, Any]:
        if body.workflow_id is None and sum(x is not None for x in (body.goal, body.workflow)) != 1:
            raise HTTPException(422, "give exactly one of 'goal' or 'workflow', or provide a 'workflow_id'")
        if body.workflow_id is not None and body.workflow is not None:
            raise HTTPException(422, "do not provide both 'workflow_id' and 'workflow'")
        svc = services()
        if body.workflow_id is not None:
            try:
                template = svc.store.get("workflows", body.workflow_id) or svc.store.get(
                    "workflow-versions", body.workflow_id
                )
            except ValueError:
                template = None
            if template is None:
                raise HTTPException(404, f"unknown workflow '{body.workflow_id}'")
            body.workflow = template["workflow"]
        available = available_capabilities(svc, who, include_writes=body.workflow is not None)
        if body.goal is not None and body.workflow_id is None:
            planned = await require_planner().plan(body.goal, body.inputs, svc.registry.capabilities, available)
            if not planned.ok or planned.spec is None:
                raise HTTPException(422, planned.view())
            spec = planned.spec
            meta: dict[str, Any] = {
                "goal": body.goal,
                "rationale": planned.rationale,
                "attempts": planned.attempts,
                "considered": planned.considered,
                "planned_by": "planner",
            }
            if warnings := getattr(planned, "warnings", None):
                meta["warnings"] = list(warnings)
        else:
            try:
                spec = WorkflowSpec.model_validate(body.workflow)
            except ValidationError as exc:
                raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
            if body.workflow_id is not None:
                if body.goal is not None:
                    spec, body.inputs = _prepare_saved_workflow(
                        spec, body.goal, body.inputs, svc.registry.capabilities
                    )
                    spec.goal = body.goal
                meta = {
                    "goal": body.goal or spec.goal,
                    "rationale": "Saved workflow selected by the user.",
                    "attempts": 0,
                    "considered": [],
                    "planned_by": "saved_workflow",
                }
                meta["workflow_id"] = body.workflow_id
            else:
                meta = {"goal": spec.goal, "rationale": "", "attempts": 0, "considered": [], "planned_by": "user"}
        meta["routing"] = routing_view(svc, who, spec, available, meta["considered"])
        capability_nodes = [n for n in spec.nodes if n.type is NodeType.CAPABILITY and n.capability]
        blocked = sorted({n.capability for n in capability_nodes if n.capability not in available})
        if blocked:
            raise HTTPException(403, f"not allowed or not available: {blocked}")
        if svc.node_timeout is not None:
            for node in capability_nodes:
                # validation needs a timeout on every capability node; raising existing ones is the runner's job
                if node.timeout_s is None or not svc.runner_applies_node_timeout:
                    minimum = int(svc.node_timeout(node.capability or "") or 0)
                    node.timeout_s = max(node.timeout_s or 0, minimum) or None
        try:
            run = await svc.runs.start(spec, body.inputs, who, plan=meta)
        except RunError as exc:
            raise HTTPException(422, str(exc)) from exc
        if body.wait_s > 0:
            with suppress(TimeoutError):
                await svc.runs.wait(run.id, timeout=body.wait_s)
        return run.view()

    @app.get("/api/v1/runs")
    def list_runs(who: Annotated[Principal, Depends(principal)]) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return [
            run.summary() for run in services().runs.list() if who.role == "admin" or run.principal.name == who.name
        ]

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(  # pyright: ignore[reportUnusedFunction]
        run_id: str,
        who: Annotated[Principal, Depends(principal)],
        wait_s: Annotated[float, Query(ge=0, le=60)] = 0,
    ) -> dict[str, Any]:
        run = owned_run(run_id, who)
        if wait_s > 0 and str(run.status) == "running":
            with suppress(TimeoutError):
                await services().runs.wait(run_id, timeout=wait_s)
        return run.view()

    @app.post("/api/v1/runs/{run_id}/inputs/{node_id}")
    async def submit_input(  # pyright: ignore[reportUnusedFunction]
        run_id: str, node_id: str, body: HumanInputRequest, who: Annotated[Principal, Depends(principal)]
    ) -> dict[str, Any]:
        owned_run(run_id, who)
        try:
            return (await services().runs.submit_input(run_id, node_id, body.value)).view()
        except RunError as exc:
            raise HTTPException(422, str(exc)) from exc

    def can_approve(who: Principal, run: Run, pending: PendingApproval) -> bool:
        entitled = who.role == "admin" or services().entitlements.allows(who.role, f"approve:{pending.capability}")
        return entitled and (settings.allow_self_approval or who.name != run.principal.name)

    @app.get("/api/v1/approvals")
    def approvals(who: Annotated[Principal, Depends(principal)]) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return [
            _approval_view(run, pending)
            for run, pending in services().runs.pending_approvals()
            if can_approve(who, run, pending)
        ]

    @app.post("/api/v1/runs/{run_id}/approvals/{node_id}")
    async def decide(  # pyright: ignore[reportUnusedFunction]
        run_id: str, node_id: str, body: DecisionRequest, who: Annotated[Principal, Depends(principal)]
    ) -> dict[str, Any]:
        runs = services().runs
        try:
            run = runs.get(run_id)
        except RunError as exc:
            raise HTTPException(404, str(exc)) from exc
        owner = who.role == "admin" or run.principal.name == who.name
        pending = next((p for p in run.approvals.values() if p.node == node_id and p.item == body.item), None)
        if pending is None:
            if not owner:
                raise HTTPException(404, f"unknown run '{run_id}'")
            raise HTTPException(422, f"node '{node_id}' has no pending approval for item {body.item}")
        if not can_approve(who, run, pending):
            if who.name == run.principal.name and not settings.allow_self_approval:
                raise HTTPException(403, "four-eyes: approver must differ from requester")
            raise HTTPException(403, f"role '{who.role}' may not approve {pending.capability}")
        try:
            run = await runs.decide(
                run_id, node_id, item=body.item, decision=body.decision, principal=who, payload_hash=body.payload_hash
            )
        except RunError as exc:
            raise HTTPException(409 if str(exc).startswith("stale approval") else 422, str(exc)) from exc
        return run.view(include_inputs=owner)

    @app.post("/api/v1/runs/{run_id}/cancel")
    async def cancel_run(  # pyright: ignore[reportUnusedFunction]
        run_id: str, who: Annotated[Principal, Depends(principal)]
    ) -> dict[str, Any]:
        owned_run(run_id, who)
        cancel = getattr(services().runs, "cancel", None)
        if cancel is None:
            raise HTTPException(501, "run cancellation is not available")
        try:
            return (await cancel(run_id)).view()
        except RunError as exc:
            raise HTTPException(422, str(exc)) from exc

    app.include_router(create_mcp_router(services, principal, allowed_origins=settings.allowed_origins))
    # feature routers: create_router(services, principal, admin, settings) -> APIRouter
    for module in (agents_api, workflows_api, skills_api, mcp_catalog_api, goals_api):
        if module is not None:
            app.include_router(module.create_router(services, principal, admin, settings))

    app.mount("/assets", StaticFiles(directory=STATIC / "assets", check_dir=False), name="assets")
    return app
