"""In-memory run controller: executes a WorkflowSpec through the gateway, concurrently where the graph allows."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import jsonschema

from ueoa.contracts import CapabilityContract, Effect, InputSpec, NodeType, WorkflowNode, WorkflowSpec
from ueoa.gateway.service import (
    Approval,
    GatewayDeniedError,
    GatewayFailedError,
    InvocationRecord,
    Principal,
)
from ueoa.gateway.service import payload_hash as hash_payload
from ueoa.persistence.writebehind import schedule_write

from .expressions import ExpressionError, evaluate_source, resolve_bindings
from .scheduler import TERMINAL, NodeState, RunStatus, aggregate_item_states, eligible_nodes, must_skip, run_status
from .validation import validate_spec

DEFAULT_ITEM_CONCURRENCY = 8
UNFINISHED = frozenset({RunStatus.RUNNING, RunStatus.WAITING})
EXPIRED_DETAIL = "expired waiting for input"
_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
_SCHEMA_KEYWORDS = frozenset(
    {"enum", "const", "items", "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern"}
    | {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "properties", "required", "format"}
    | {"additionalProperties"}
)
_ERROR_CODE_ALIASES = {"output schema": "bad_output"}
_PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*(?:\.\w+)*)\}")
logger = logging.getLogger(__name__)


def _input_schema(declared: InputSpec) -> dict[str, Any]:
    """JSON Schema for a declared workflow input; unknown scalar types are left unconstrained."""
    kind = declared.type
    if kind in _JSON_TYPES:
        schema: dict[str, Any] = {"type": kind}
    elif kind == "string_list":
        schema = {"type": "array", "items": {"type": "string"}}
    elif kind == "list" or kind.endswith("_list"):
        schema = {"type": "array"}
    else:
        schema = {}
    extras = declared.model_extra or {}
    schema.update({key: value for key, value in extras.items() if key in _SCHEMA_KEYWORDS})
    return schema


def _error_code(message: str) -> str:
    prefix = message.split(":", 1)[0].strip()
    return _ERROR_CODE_ALIASES.get(prefix, prefix)


class RunError(ValueError):
    """Invalid spec or inputs, unknown run, bad human input, or node not waiting."""


class InvocationGateway(Protocol):
    async def invoke(
        self, principal: Principal, capability_id: str, payload: dict[str, Any], *, approval: Approval | None = None
    ) -> InvocationRecord: ...


def _fill_template(template: str, context: dict[str, Any]) -> str:
    """Replace {a.b} placeholders with values from context; unknown placeholders raise KeyError."""

    def lookup(match: re.Match[str]) -> str:
        value: Any = context
        for part in match.group(1).split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(match.group(1))
            value = value[part]  # pyright: ignore[reportUnknownVariableType]
        return str(value)  # pyright: ignore[reportUnknownArgumentType]

    return _PLACEHOLDER.sub(lookup, template)


def _approval_key(node_id: str, item: int | None) -> str:
    return node_id if item is None else f"{node_id}[{item}]"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunEvent:
    at: str
    type: str
    node: str | None = None
    item: int | None = None
    detail: str = ""
    invocation_id: str | None = None
    implementation: str | None = None


@dataclass
class ItemResult:
    index: int
    input: Any
    state: NodeState
    output: dict[str, Any] | None = None
    error: str | None = None
    invocation_id: str | None = None
    implementation: str | None = None
    attempts: int = 0


@dataclass
class NodeResult:
    state: NodeState
    output: dict[str, Any] | None = None
    items: list[ItemResult] = field(default_factory=lambda: [])
    error: str | None = None
    invocation_id: str | None = None
    implementation: str | None = None
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    timeout_s: float | None = None


@dataclass
class PendingApproval:
    node: str
    item: int | None
    capability: str
    payload: dict[str, Any]
    payload_hash: str
    idempotency_key: str
    requested_at: str
    requested_by: str


@dataclass(kw_only=True)
class Run:
    id: str
    spec: WorkflowSpec
    inputs: dict[str, Any]
    principal: Principal
    status: RunStatus
    nodes: dict[str, NodeResult]
    events: list[RunEvent]
    outputs: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    finished_at: str | None = None
    plan: dict[str, Any] | None = None
    approvals: dict[str, PendingApproval] = field(default_factory=lambda: {})

    def states(self) -> dict[str, NodeState]:
        return {node_id: result.state for node_id, result in self.nodes.items()}

    def emit(self, type_: str, node: str | None = None, **details: Any) -> None:
        self.events.append(RunEvent(_now(), type_, node, **details))

    def view(self, include_inputs: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": str(self.status),
            "goal": self.spec.goal,
            "inputs": self.inputs if include_inputs else None,
            "inputs_summary": {name: len(json.dumps(value, default=str)) for name, value in self.inputs.items()},
            "plan": self.plan,
            "workflow": self.spec.model_dump(mode="json", exclude_none=True),
            "nodes": {node_id: _node_view(result) for node_id, result in self.nodes.items()},
            "approvals": [asdict(pending) for pending in self.approvals.values()],
            "events": [asdict(event) for event in self.events],
            "outputs": self.outputs,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": str(self.status),
            "goal": self.spec.goal,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "nodes": dict(Counter(str(result.state) for result in self.nodes.values())),
        }


def _node_view(result: NodeResult) -> dict[str, Any]:
    view = asdict(result)
    view["state"] = str(result.state)
    for item in view["items"]:
        item["state"] = str(item["state"])
    return view


class RunController:
    def __init__(
        self,
        gateway: InvocationGateway,
        capabilities: Callable[[], Mapping[str, CapabilityContract]],
        *,
        default_timeout_s: float = 120.0,
        max_backoff_s: float = 30.0,
        max_runs: int = 200,
        max_items_per_node: int = 50,
        max_parallel_invocations: int = 8,
        node_timeout: Callable[[str], float | None] | None = None,
        waiting_ttl_s: float = 3600,
        max_active_runs: int = 20,
        max_active_runs_per_principal: int = 5,
        max_total_invocations: int = 32,
        persist_run: Callable[[Run], Coroutine[Any, Any, None]] | None = None,
        persist_approval: Callable[[str, str, int | None, str, str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._gateway = gateway
        self._capabilities = capabilities
        self._default_timeout_s = default_timeout_s
        self._max_backoff_s = max_backoff_s
        self._max_runs = max_runs
        self._max_items = max_items_per_node
        self._max_parallel = max_parallel_invocations
        self._node_timeout = node_timeout
        self._waiting_ttl_s = waiting_ttl_s
        self._max_active = max_active_runs
        self._max_active_per_principal = max_active_runs_per_principal
        self._total_limiter = asyncio.Semaphore(max_total_invocations)
        self._runs: dict[str, Run] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._limiters: dict[str, asyncio.Semaphore] = {}
        self._wake: dict[str, asyncio.Event] = {}
        self._waiting_since: dict[str, float] = {}
        self._persist = persist_run
        self._persist_approval = persist_approval

    def restore(self, records: list[dict[str, Any]]) -> None:
        for record in reversed(records):
            try:
                run = self._restore_run(record)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("stored run skipped: %s", exc)
                continue
            self._runs[run.id] = run
            self._limiters[run.id] = asyncio.Semaphore(self._max_parallel)
            self._wake[run.id] = asyncio.Event()

    @staticmethod
    def _restore_run(record: dict[str, Any]) -> Run:
        spec = WorkflowSpec.model_validate(record["workflow"])
        nodes: dict[str, NodeResult] = {}
        for node in spec.nodes:
            raw = record.get("nodes", {}).get(node.id, {})
            items = [
                ItemResult(
                    index=int(item["index"]),
                    input=item.get("input"),
                    state=NodeState(item["state"]),
                    output=item.get("output"),
                    error=item.get("error"),
                    invocation_id=item.get("invocation_id"),
                    implementation=item.get("implementation"),
                    attempts=int(item.get("attempts", 0)),
                )
                for item in raw.get("items", [])
            ]
            nodes[node.id] = NodeResult(
                state=NodeState(raw.get("state", NodeState.PENDING)),
                output=raw.get("output"),
                items=items,
                error=raw.get("error"),
                invocation_id=raw.get("invocation_id"),
                implementation=raw.get("implementation"),
                attempts=int(raw.get("attempts", 0)),
                started_at=raw.get("started_at"),
                finished_at=raw.get("finished_at"),
                timeout_s=raw.get("timeout_s", node.timeout_s),
            )
        approvals = {
            _approval_key(str(item["node"]), item.get("item")): PendingApproval(
                node=str(item["node"]),
                item=item.get("item"),
                capability=str(item["capability"]),
                payload=item["payload"],
                payload_hash=str(item["payload_hash"]),
                idempotency_key=str(item["idempotency_key"]),
                requested_at=str(item["requested_at"]),
                requested_by=str(item["requested_by"]),
            )
            for item in record.get("approvals", [])
        }
        return Run(
            id=str(record["id"]),
            spec=spec,
            inputs=record.get("inputs", {}),
            principal=Principal(str(record["principal"]), str(record["role"])),
            status=RunStatus(record["status"]),
            nodes=nodes,
            events=[],
            outputs=record.get("outputs"),
            error=record.get("error"),
            created_at=str(record["created_at"]),
            finished_at=record.get("finished_at"),
            plan=record.get("plan"),
            approvals=approvals,
        )

    def _schedule_persist(self, run: Run) -> None:
        if self._persist is not None:
            schedule_write(self._persist(copy.deepcopy(run)), f"run {run.id}")

    async def start(
        self, spec: WorkflowSpec, inputs: dict[str, Any], principal: Principal, *, plan: dict[str, Any] | None = None
    ) -> Run:
        self._expire()
        active = [run for run in self._runs.values() if run.finished_at is None]
        if len(active) >= self._max_active:
            raise RunError(f"too many active runs; the limit is {self._max_active}")
        if sum(1 for run in active if run.principal.name == principal.name) >= self._max_active_per_principal:
            raise RunError(
                f"too many active runs for '{principal.name}'; the limit is {self._max_active_per_principal}"
            )
        issues = validate_spec(spec, self._capabilities())
        if issues:
            raise RunError("invalid workflow: " + "; ".join(str(issue) for issue in issues))
        resolved = dict(inputs)
        missing: list[str] = []
        problems: list[str] = []
        for name, declared in spec.inputs.items():
            if name in resolved:
                problems.extend(_input_problems(name, declared, resolved[name]))
                continue
            if "default" in declared.model_fields_set:
                resolved[name] = copy.deepcopy(declared.default)
            else:
                missing.append(name)
        if missing:
            raise RunError(f"missing required inputs: {', '.join(missing)}")
        if problems:
            raise RunError("invalid inputs: " + "; ".join(problems))

        self._evict()
        run = Run(
            id=str(uuid.uuid4()),
            spec=spec,
            inputs=resolved,
            principal=principal,
            status=RunStatus.RUNNING,
            nodes={node.id: NodeResult(NodeState.PENDING, timeout_s=self._timeout(node)) for node in spec.nodes},
            events=[],
            created_at=_now(),
            plan=plan,
        )
        self._runs[run.id] = run
        self._limiters[run.id] = asyncio.Semaphore(self._max_parallel)
        self._wake[run.id] = asyncio.Event()
        run.emit("run_started", detail=spec.goal)
        self._schedule_persist(run)
        self._launch(run)
        return run

    async def execute(
        self, spec: WorkflowSpec, inputs: dict[str, Any], principal: Principal, *, plan: dict[str, Any] | None = None
    ) -> Run:
        run = await self.start(spec, inputs, principal, plan=plan)
        return await self.wait(run.id)

    def get(self, run_id: str) -> Run:
        self._expire()
        run = self._runs.get(run_id)
        if run is None:
            raise RunError(f"unknown run '{run_id}'")
        return run

    def list(self) -> list[Run]:
        self._expire()
        return list(reversed(self._runs.values()))

    async def wait(self, run_id: str, timeout: float | None = None) -> Run:
        run = self.get(run_id)
        async with asyncio.timeout(timeout):
            while (task := self._tasks.get(run_id)) is not None and not task.done():
                await asyncio.wait({task})
        return run

    async def cancel(self, run_id: str) -> Run:
        return await self._cancel(self.get(run_id), "cancelled by request")

    async def close(self) -> None:
        for run in [run for run in self._runs.values() if run.finished_at is None]:
            if self._persist is not None and run.status is RunStatus.WAITING:
                self._schedule_persist(run)
                continue
            await self._cancel(run, "controller closed")

    async def submit_input(self, run_id: str, node_id: str, value: dict[str, Any]) -> Run:
        run = self.get(run_id)
        result = run.nodes.get(node_id)
        if result is None:
            raise RunError(f"run '{run_id}' has no node '{node_id}'")
        if result.state is not NodeState.WAITING_INPUT:
            raise RunError(f"node '{node_id}' is not waiting for input (state {result.state})")
        if not isinstance(value, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise RunError("input must be a JSON object")
        try:
            jsonschema.validate(value, run.spec.node(node_id).input_schema or {})
        except jsonschema.ValidationError as exc:
            raise RunError(f"input rejected: {exc.message}") from exc
        result.state, result.output, result.finished_at = NodeState.SUCCEEDED, value, _now()
        run.emit("input_received", node_id)
        run.emit("node_succeeded", node_id)
        self._schedule_persist(run)
        self._resume(run)
        return run

    def pending_approvals(self) -> list[tuple[Run, PendingApproval]]:
        self._expire()
        return [(run, pending) for run in self._runs.values() for pending in run.approvals.values()]

    async def decide(
        self,
        run_id: str,
        node_id: str,
        *,
        item: int | None,
        decision: Literal["approve", "reject"],
        principal: Principal,
        payload_hash: str,
    ) -> Run:
        run = self.get(run_id)
        if node_id not in run.nodes:
            raise RunError(f"run '{run_id}' has no node '{node_id}'")
        key = _approval_key(node_id, item)
        pending = run.approvals.get(key)
        if pending is None:
            raise RunError(f"'{key}' is not waiting for approval")
        if payload_hash != pending.payload_hash:
            raise RunError("stale approval: the payload hash does not match the pending payload")
        node = run.spec.node(node_id)
        result = run.nodes[node_id]
        target: NodeResult | ItemResult = result if item is None else result.items[item]
        del run.approvals[key]
        if decision == "reject":
            target.state, target.error = NodeState.REJECTED, f"rejected by {principal.name}"
            run.emit("approval_rejected", node_id, item=item, detail=principal.name)
        else:
            approval = Approval(
                str(uuid.uuid4()), pending.payload_hash, principal.name, _now(), pending.idempotency_key
            )
            target.state = NodeState.RUNNING
            run.status = RunStatus.RUNNING
            self._waiting_since.pop(run_id, None)
            run.emit("approval_granted", node_id, item=item, detail=f"{principal.name} ({approval.approval_id})")
            try:
                await self._invoke(run, node, pending.payload, target, item, approval)
            except RunError as exc:
                target.state = NodeState.WAITING_APPROVAL
                run.approvals[key] = pending
                run.emit("approval_refused", node_id, item=item, detail=str(exc))
                if run.finished_at is None:
                    self._finish(run)
                raise
        if item is not None:
            kind = {NodeState.SUCCEEDED: "item_succeeded", NodeState.REJECTED: "item_rejected"}.get(
                target.state, "item_failed"
            )
            run.emit(kind, node_id, item=item, detail=target.error or "", **_ids(target))
        if run.finished_at is None:
            if node.for_each is not None:
                self._aggregate(node, result)
            self._settle(run, node)
            self._resume(run)
        self._schedule_persist(run)
        if self._persist_approval is not None:
            schedule_write(
                self._persist_approval(run_id, node_id, item, decision, principal.name),
                f"approval decision {run_id}/{node_id}",
            )
        return run

    # ------------------------------------------------------------ internals

    def _resume(self, run: Run) -> None:
        run.status = RunStatus.RUNNING
        self._waiting_since.pop(run.id, None)
        task = self._tasks.get(run.id)
        if task is None or task.done():
            self._launch(run)
        else:
            self._wake[run.id].set()

    def _needs_approval(self, node: WorkflowNode) -> bool:
        capability = self._capabilities().get(node.capability or "")
        return node.effect is Effect.WRITE or (capability is not None and capability.effects is Effect.WRITE)

    def _request_approval(
        self, run: Run, node: WorkflowNode, payload: dict[str, Any], target: NodeResult | ItemResult, index: int | None
    ) -> None:
        digest = hash_payload(payload)
        position = index or 0
        element = target.input if isinstance(target, ItemResult) else None
        item: dict[str, Any] = dict(element) if isinstance(element, dict) else {"value": element}  # pyright: ignore[reportUnknownArgumentType]
        item.update(payload_hash=digest, index=position)
        if node.idempotency_key:
            context = {"run": {"id": run.id}, "node": {"id": node.id}, "item": item}
            try:
                key = _fill_template(node.idempotency_key, context)
            except KeyError as exc:
                target.state, target.error = NodeState.FAILED, f"idempotency_key: unknown placeholder {{{exc.args[0]}}}"
                return
        else:
            key = f"{run.id}:{node.id}:{position}:{digest}"
        target.state = NodeState.WAITING_APPROVAL
        run.approvals[_approval_key(node.id, index)] = PendingApproval(
            node.id, index, node.capability or "", payload, digest, key, _now(), run.principal.name
        )
        run.emit("approval_requested", node.id, item=index, detail=digest)
        self._schedule_persist(run)

    def _aggregate(self, node: WorkflowNode, result: NodeResult) -> None:
        states = [item.state for item in result.items]
        state = aggregate_item_states(states)
        keep_succeeded = node.on_item_error == "continue" and NodeState.SUCCEEDED in states
        if state is not NodeState.WAITING_APPROVAL and keep_succeeded:
            state = NodeState.SUCCEEDED
        result.state = state
        result.attempts = sum(item.attempts for item in result.items)
        failed = next((item for item in result.items if item.state is NodeState.FAILED), None)
        if failed is not None and result.state is NodeState.FAILED:
            result.error = f"item {failed.index}: {failed.error}"

    def _settle(self, run: Run, node: WorkflowNode) -> None:
        """Close a node that reached a terminal state; waiting or still-running nodes are left alone."""
        result = run.nodes[node.id]
        if result.state not in TERMINAL:
            return
        result.finished_at = _now()
        if result.state is NodeState.FAILED:
            run.emit("node_failed", node.id, detail=result.error or "", **_ids(result))
        elif result.state is NodeState.REJECTED:
            run.emit("node_rejected", node.id, detail=result.error or "", **_ids(result))
        else:
            failed = sum(1 for item in result.items if item.state is NodeState.FAILED)
            rejected = sum(1 for item in result.items if item.state is NodeState.REJECTED)
            parts = [f"{failed} of {len(result.items)} items failed"] if failed else []
            parts += [f"{rejected} of {len(result.items)} items rejected"] if rejected else []
            run.emit("node_succeeded", node.id, detail="; ".join(parts), **_ids(result))
        self._schedule_persist(run)

    def _timeout(self, node: WorkflowNode) -> float | None:
        if node.type is not NodeType.CAPABILITY:
            return None
        floor = (self._node_timeout(node.capability or "") if self._node_timeout else None) or 0
        return max(node.timeout_s or self._default_timeout_s, floor)

    def _expire(self) -> None:
        deadline = time.monotonic() - self._waiting_ttl_s
        for run_id, since in list(self._waiting_since.items()):
            run = self._runs.get(run_id)
            if run is not None and run.status is RunStatus.WAITING and since < deadline:
                self._mark_cancelled(run, EXPIRED_DETAIL)

    async def _cancel(self, run: Run, detail: str) -> Run:
        if run.finished_at is not None:
            return run
        task = self._tasks.get(run.id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})
        if run.finished_at is None:
            self._mark_cancelled(run, detail)
        return run

    def _mark_cancelled(self, run: Run, detail: str) -> None:
        if self._persist_approval is not None:
            for pending in run.approvals.values():
                schedule_write(
                    self._persist_approval(run.id, pending.node, pending.item, "cancelled", "system"),
                    f"approval cancellation {run.id}/{pending.node}",
                )
        self._cancel_open_nodes(run)
        run.status, run.error, run.finished_at = RunStatus.CANCELLED, detail, _now()
        self._waiting_since.pop(run.id, None)
        run.emit("run_cancelled", detail=detail)
        self._schedule_persist(run)

    def _cancel_open_nodes(self, run: Run) -> None:
        now = _now()
        run.approvals.clear()
        for result in run.nodes.values():
            for item in result.items:
                if item.state not in TERMINAL:
                    item.state = NodeState.CANCELLED
            if result.state not in TERMINAL:
                result.state, result.finished_at = NodeState.CANCELLED, now

    def _evict(self) -> None:
        excess = len(self._runs) - self._max_runs + 1
        for run_id in [rid for rid, run in self._runs.items() if run.finished_at is not None][: max(excess, 0)]:
            del self._runs[run_id]
            self._tasks.pop(run_id, None)
            self._limiters.pop(run_id, None)
            self._wake.pop(run_id, None)
            self._waiting_since.pop(run_id, None)

    def _launch(self, run: Run) -> None:
        self._tasks[run.id] = asyncio.create_task(self._drive(run), name=f"run-{run.id}")

    async def _drive(self, run: Run) -> None:
        running: dict[asyncio.Task[None], str] = {}
        wake = self._wake.setdefault(run.id, asyncio.Event())
        try:
            while True:
                wake.clear()
                while eligible := eligible_nodes(run.spec, run.states()):
                    for node_id in eligible:
                        task = self._schedule(run, run.spec.node(node_id))
                        if task is not None:
                            running[task] = node_id
                if not running:
                    break
                waker = asyncio.create_task(wake.wait())
                try:
                    done, _ = await asyncio.wait({*running, waker}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    waker.cancel()
                for task in done:
                    running.pop(task, None)
        except asyncio.CancelledError:
            await _stop(running)
            raise
        except Exception as exc:  # the loop itself must never leave a run hanging
            await _stop(running)
            self._cancel_open_nodes(run)
            run.status, run.error, run.finished_at = RunStatus.FAILED, f"{type(exc).__name__}: {exc}", _now()
            run.emit("run_finished", detail=run.error)
            self._schedule_persist(run)
            return
        self._finish(run)

    def _schedule(self, run: Run, node: WorkflowNode) -> asyncio.Task[None] | None:
        result = run.nodes[node.id]
        if must_skip(run.spec, node.id, run.states()):
            self._skip(run, node.id, "a dependency was skipped")
            return None
        if node.condition is not None:
            try:
                verdict = evaluate_source(node.condition, self._context(run, node))
            except ExpressionError as exc:
                self._fail(run, node.id, f"condition: {exc}")
                return None
            if not isinstance(verdict, bool):
                self._fail(run, node.id, f"condition must be a boolean, got {type(verdict).__name__}")
                return None
            if not verdict:
                self._skip(run, node.id, "condition is false")
                return None
        result.started_at = _now()
        if node.type is NodeType.HUMAN_INPUT:
            result.state = NodeState.WAITING_INPUT
            run.emit("node_waiting_input", node.id, detail=node.prompt_key or "")
            self._schedule_persist(run)
            return None
        result.state = NodeState.RUNNING
        run.emit("node_started", node.id, detail=node.capability or "")
        self._schedule_persist(run)
        return asyncio.create_task(self._run_node(run, node))

    def _skip(self, run: Run, node_id: str, reason: str) -> None:
        result = run.nodes[node_id]
        result.state, result.finished_at = NodeState.SKIPPED, _now()
        run.emit("node_skipped", node_id, detail=reason)
        self._schedule_persist(run)

    def _fail(self, run: Run, node_id: str, error: str) -> None:
        result = run.nodes[node_id]
        result.state, result.error, result.finished_at = NodeState.FAILED, error, _now()
        run.emit("node_failed", node_id, detail=error)
        self._schedule_persist(run)

    def _context(self, run: Run, node: WorkflowNode | None = None) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        for node_id, result in run.nodes.items():
            if result.state is not NodeState.SUCCEEDED:
                continue
            if run.spec.node(node_id).for_each is None:
                nodes[node_id] = {"output": result.output}
            else:
                nodes[node_id] = {
                    "items": [
                        {"input": item.input, "output": item.output}
                        for item in result.items
                        if item.state is NodeState.SUCCEEDED
                    ],
                    "failed": [
                        {"index": item.index, "input": item.input, "error": item.error}
                        for item in result.items
                        if item.state is NodeState.FAILED
                    ],
                }
        context: dict[str, Any] = {
            "inputs": run.inputs,
            "nodes": nodes,
            "run": {"id": run.id, "locale": run.inputs.get("locale", "en")},
        }
        if node is not None:
            context["node"] = {"id": node.id}
        return context

    async def _run_node(self, run: Run, node: WorkflowNode) -> None:
        result = run.nodes[node.id]
        try:
            context = self._context(run, node)
            if node.for_each is None:
                try:
                    payload = resolve_bindings(node.inputs, context)
                except ExpressionError as exc:
                    self._fail(run, node.id, f"input binding: {exc}")
                    return
                if self._needs_approval(node):
                    self._request_approval(run, node, payload, result, None)
                else:
                    await self._invoke(run, node, payload, result, None)
            else:
                try:
                    elements = evaluate_source(node.for_each, context)
                except ExpressionError as exc:
                    self._fail(run, node.id, f"for_each: {exc}")
                    return
                if not isinstance(elements, list):
                    self._fail(run, node.id, f"for_each must produce a list, got {type(elements).__name__}")
                    return
                if len(elements) > self._max_items:  # pyright: ignore[reportUnknownArgumentType]
                    self._fail(
                        run,
                        node.id,
                        f"for_each produced {len(elements)} items; the limit is {self._max_items} per node",  # pyright: ignore[reportUnknownArgumentType]
                    )
                    return
                await self._fan_out(run, node, context, elements)  # pyright: ignore[reportUnknownArgumentType]
        except Exception as exc:  # never crash the run loop
            self._fail(run, node.id, f"{type(exc).__name__}: {exc}")
            return
        if result.state is NodeState.WAITING_APPROVAL:
            pending = sum(1 for pending in run.approvals.values() if pending.node == node.id)
            run.emit("node_waiting_approval", node.id, detail=f"{pending} approval(s) pending")
            return
        self._settle(run, node)

    async def _fan_out(self, run: Run, node: WorkflowNode, context: dict[str, Any], elements: list[Any]) -> None:
        result = run.nodes[node.id]
        result.items = [ItemResult(index, element, NodeState.PENDING) for index, element in enumerate(elements)]
        semaphore = asyncio.Semaphore(node.max_concurrency or DEFAULT_ITEM_CONCURRENCY)

        async def run_item(item: ItemResult) -> None:
            async with semaphore:
                item.state = NodeState.RUNNING
                run.emit("item_started", node.id, item=item.index)
                try:
                    payload = resolve_bindings(node.inputs, {**context, "item": item.input})
                except ExpressionError as exc:
                    item.state, item.error = NodeState.FAILED, f"input binding: {exc}"
                else:
                    if self._needs_approval(node):
                        self._request_approval(run, node, payload, item, item.index)
                    else:
                        await self._invoke(run, node, payload, item, item.index)
                if item.state is NodeState.WAITING_APPROVAL:
                    return
                kind = "item_succeeded" if item.state is NodeState.SUCCEEDED else "item_failed"
                run.emit(kind, node.id, item=item.index, detail=item.error or "", **_ids(item))

        await asyncio.gather(*(run_item(item) for item in result.items))
        self._aggregate(node, result)

    async def _invoke(
        self,
        run: Run,
        node: WorkflowNode,
        payload: dict[str, Any],
        target: NodeResult | ItemResult,
        item: int | None,
        approval: Approval | None = None,
    ) -> None:
        timeout = run.nodes[node.id].timeout_s or self._default_timeout_s
        retry = node.retry
        limiter = self._limiters.get(run.id) or asyncio.Semaphore(self._max_parallel)
        capability = node.capability or ""
        while True:
            target.attempts += 1
            try:
                async with limiter, self._total_limiter:
                    call = (
                        self._gateway.invoke(run.principal, capability, payload, approval=approval)
                        if approval is not None
                        else self._gateway.invoke(run.principal, capability, payload)
                    )
                    record = await asyncio.wait_for(call, timeout)
            except GatewayDeniedError as exc:
                _note(target, exc.record)
                step = exc.record.steps[-1].name if exc.record.steps else "gateway"
                if approval is not None and step == "effect":
                    target.attempts -= 1  # the approval was refused; the write was never sent
                    raise RunError(f"approval refused: {exc.record.error or exc}") from exc
                target.state, target.error = NodeState.FAILED, f"denied at {step}: {exc.record.error or exc}"
                return
            except (GatewayFailedError, TimeoutError) as exc:
                if isinstance(exc, GatewayFailedError):
                    _note(target, exc.record)
                    message = exc.record.error or str(exc) or "failed"
                else:
                    message = f"timeout: exceeded {timeout}s"
                code = getattr(exc, "code", None) or _error_code(message)
                if approval is None and code in retry.retry_on and target.attempts < retry.max_attempts:
                    index = target.attempts - 1
                    delay = min(retry.backoff_s[index] if index < len(retry.backoff_s) else 0.0, self._max_backoff_s)
                    detail = f"attempt {target.attempts} failed ({message}); retrying in {delay}s"
                    run.emit("item_retry", node.id, item=item, detail=detail)
                    await asyncio.sleep(delay)
                    continue
                target.state, target.error = NodeState.FAILED, message
                return
            except Exception as exc:
                target.state, target.error = NodeState.FAILED, f"{type(exc).__name__}: {exc}"
                return
            _note(target, record)
            target.state, target.output, target.error = NodeState.SUCCEEDED, record.output, None
            return

    def _finish(self, run: Run) -> None:
        spec = run.spec
        run.status = run_status(spec, run.states())
        if run.status is RunStatus.WAITING:
            self._waiting_since[run.id] = time.monotonic()
        if run.status in UNFINISHED:
            self._schedule_persist(run)
            return
        context = self._context(run)
        outputs: dict[str, Any] = {}
        if spec.outputs:
            for name, source in spec.outputs.items():
                try:
                    outputs[name] = evaluate_source(source, context)
                except ExpressionError as exc:
                    outputs[name] = None
                    run.emit("output_failed", detail=f"{name}: {exc}")
        else:
            depended = {dep for node in spec.nodes for dep in node.depends_on}
            for node in spec.nodes:
                result = run.nodes[node.id]
                if node.id in depended or result.state is not NodeState.SUCCEEDED:
                    continue
                outputs[node.id] = (
                    result.output
                    if node.for_each is None
                    else [i.output for i in result.items if i.state is NodeState.SUCCEEDED]
                )
        run.outputs = outputs
        if run.status is RunStatus.FAILED:
            failed = next((nid for nid, r in run.nodes.items() if r.state is NodeState.FAILED), None)
            run.error = f"{failed}: {run.nodes[failed].error}" if failed else "workflow blocked"
        run.finished_at = _now()
        run.emit("run_finished", detail=str(run.status) if run.error is None else f"{run.status}: {run.error}")
        self._schedule_persist(run)


def _input_problems(name: str, declared: InputSpec, value: Any) -> list[str]:
    schema = _input_schema(declared)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        return [f"{name}: invalid declaration ({exc.message})"]
    return [f"{name}: {error.message}" for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]


async def _stop(tasks: Mapping[asyncio.Task[None], str]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _note(target: NodeResult | ItemResult, record: InvocationRecord) -> None:
    target.invocation_id = record.invocation_id
    target.implementation = record.implementation or target.implementation


def _ids(target: NodeResult | ItemResult) -> dict[str, str | None]:
    return {"invocation_id": target.invocation_id, "implementation": target.implementation}
