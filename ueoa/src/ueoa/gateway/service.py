"""Invocation gateway: the only path to any implementation. Every decision is recorded as a step."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NoReturn, Protocol

import jsonschema

from ueoa.contracts import Effect
from ueoa.persistence.writebehind import schedule_write
from ueoa.registry.service import Registry, RegistryError

from .invokers import InvocationError, Invoker

MAX_REPLAYS = 1000


class AuditWriter(Protocol):
    async def append(self, entry: dict[str, Any]) -> None: ...


def payload_hash(payload: Any) -> str:
    """sha256 of the canonical JSON form; approvals bind to this exact value."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    name: str
    role: str


@dataclass(frozen=True)
class Approval:
    approval_id: str
    payload_hash: str
    approved_by: str  # principal name
    approved_at: str
    idempotency_key: str | None = None


@dataclass
class Step:
    name: str
    outcome: str  # pass | deny | error
    detail: str


@dataclass
class InvocationRecord:
    invocation_id: str
    at: str
    principal: Principal
    capability: str
    status: str  # succeeded | denied | failed
    implementation: str | None = None
    duration_ms: int = 0
    steps: list[Step] = field(default_factory=lambda: [])
    output: dict[str, Any] | None = None
    error: str | None = None


class GatewayDeniedError(PermissionError):
    def __init__(self, record: InvocationRecord) -> None:
        super().__init__(record.error or "denied")
        self.record = record


class GatewayFailedError(RuntimeError):
    def __init__(self, record: InvocationRecord, code: str = "remote_error") -> None:
        super().__init__(record.error or "failed")
        self.record = record
        self.code = code  # InvocationError codes, plus bad_output


class Entitlements:
    def __init__(self, roles: dict[str, list[str]]) -> None:
        self._roles = roles

    def roles(self) -> list[str]:
        return sorted(self._roles)

    def allows(self, role: str, capability_id: str) -> bool:
        return any(fnmatch.fnmatchcase(capability_id, pattern) for pattern in self._roles.get(role, []))


class Gateway:
    def __init__(
        self,
        registry: Registry,
        invoker: Invoker,
        entitlements: Entitlements,
        *,
        max_audit: int = 2000,
        allow_self_approval: bool = False,
        audit_writer: AuditWriter | None = None,
    ) -> None:
        self._registry = registry
        self._invoker = invoker
        self._entitlements = entitlements
        self._allow_self_approval = allow_self_approval
        self._audit_writer = audit_writer
        self._replays: OrderedDict[tuple[str, str], tuple[str | None, dict[str, Any]]] = OrderedDict()
        self.audit: deque[InvocationRecord] = deque(maxlen=max_audit)

    def available(self, principal: Principal, *, include_writes: bool = False) -> list[str]:
        """Capabilities this principal may call now: entitled, approved implementation, read-only unless asked."""
        found: list[str] = []
        for capability_id, capability in self._registry.capabilities.items():
            if not self._entitlements.allows(principal.role, capability_id):
                continue
            if capability.effects is Effect.WRITE and not include_writes:
                continue
            try:
                self._registry.resolve(capability_id)
            except RegistryError:
                continue
            found.append(capability_id)
        return sorted(found)

    async def invoke(
        self, principal: Principal, capability_id: str, payload: dict[str, Any], *, approval: Approval | None = None
    ) -> InvocationRecord:
        record = InvocationRecord(
            invocation_id=str(uuid.uuid4()),
            at=datetime.now(UTC).isoformat(timespec="seconds"),
            principal=principal,
            capability=capability_id,
            status="failed",
        )
        started = time.perf_counter()
        try:
            output = await self._run(record, principal, capability_id, payload, approval)
            record.status, record.output = "succeeded", output
            return record
        except GatewayDeniedError:
            record.status = "denied"
            raise
        finally:
            record.duration_ms = int((time.perf_counter() - started) * 1000)
            self.audit.append(record)
            if self._audit_writer is not None:
                schedule_write(
                    self._audit_writer.append(
                        {
                            "invocation_id": record.invocation_id,
                            "at": record.at,
                            "principal": record.principal.name,
                            "role": record.principal.role,
                            "capability": record.capability,
                            "implementation": record.implementation,
                            "status": record.status,
                            "reason": record.error,
                            "duration_ms": record.duration_ms,
                            "checks": [
                                {"name": step.name, "outcome": step.outcome, "detail": step.detail}
                                for step in record.steps
                            ],
                            "output": record.output,
                        }
                    ),
                    f"audit {record.invocation_id}",
                )

    async def _run(
        self,
        record: InvocationRecord,
        principal: Principal,
        capability_id: str,
        payload: dict[str, Any],
        approval: Approval | None,
    ) -> dict[str, Any]:
        steps = record.steps
        steps.append(Step("identity", "pass", f"{principal.name} (role: {principal.role}, local demo principal)"))

        capability = self._registry.capabilities.get(capability_id)
        if capability is None:
            return self._deny(record, "capability", f"'{capability_id}' is not a registered capability")

        if not self._entitlements.allows(principal.role, capability_id):
            return self._deny(record, "entitlement", f"role '{principal.role}' may not use {capability_id}")
        steps.append(Step("entitlement", "pass", f"role '{principal.role}' may use {capability_id}"))

        replay_key: tuple[str, str] | None = None
        if capability.effects is Effect.WRITE:
            if approval is None:
                return self._deny(record, "effect", "write capability requires a human approval bound to this payload")
            if approval.payload_hash != payload_hash(payload):
                return self._deny(record, "effect", "approval does not match this payload")
            if approval.approved_by == principal.name and not self._allow_self_approval:
                return self._deny(record, "effect", "four-eyes: approver must differ from requester")
            steps.append(Step("effect", "pass", f"write approved by {approval.approved_by} ({approval.approval_id})"))
            if approval.idempotency_key:
                replay_key = (capability_id, approval.idempotency_key)
                if (stored := self._replays.get(replay_key)) is not None:
                    self._replays.move_to_end(replay_key)
                    record.implementation = stored[0]
                    steps.append(Step("idempotent_replay", "pass", f"idempotent replay of {approval.idempotency_key}"))
                    return stored[1]
        else:
            steps.append(Step("effect", "pass", "read-only capability"))

        try:
            impl = self._registry.resolve(capability_id)
        except RegistryError as exc:
            return self._deny(record, "resolve", str(exc))
        record.implementation = impl.key
        steps.append(
            Step(
                "resolve",
                "pass",
                f"{impl.key} (status {impl.status}, priority {impl.priority}, binding {impl.card.ueoa.binding})",
            )
        )

        try:
            jsonschema.validate(payload, capability.input_schema)
        except jsonschema.ValidationError as exc:
            return self._deny(record, "input_schema", exc.message)
        steps.append(Step("input_schema", "pass", "input matches the capability contract"))

        try:
            output = await self._invoker.invoke(impl.card, capability_id, payload, record.invocation_id)
        except InvocationError as exc:
            steps.append(Step("invoke", "error", f"{exc.code}: {exc}"))
            record.error = f"{exc.code}: {exc}"
            raise GatewayFailedError(record, exc.code) from exc
        steps.append(Step("invoke", "pass", f"{impl.card.ueoa.binding} call to {impl.key}"))

        try:
            jsonschema.validate(output, capability.output_schema)
        except jsonschema.ValidationError as exc:
            steps.append(Step("output_schema", "error", exc.message))
            record.error = f"bad_output: output schema: {exc.message}"
            raise GatewayFailedError(record, "bad_output") from exc
        steps.append(Step("output_schema", "pass", "output matches the capability contract"))
        if replay_key is not None:
            self._replays[replay_key] = (record.implementation, output)
            if len(self._replays) > MAX_REPLAYS:
                self._replays.popitem(last=False)
        steps.append(Step("audit", "pass", f"recorded invocation {record.invocation_id}"))
        return output

    @staticmethod
    def _deny(record: InvocationRecord, step: str, reason: str) -> NoReturn:
        record.steps.append(Step(step, "deny", reason))
        record.error = reason
        raise GatewayDeniedError(record)
