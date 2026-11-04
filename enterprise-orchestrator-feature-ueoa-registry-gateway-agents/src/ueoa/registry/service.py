"""In-memory registry: capabilities (contracts) + implementations (cards), conformance, approval, resolution.

Postgres persistence replaces the dicts later behind the same methods.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import httpx
import jsonschema
from pydantic import ValidationError

from ueoa.contracts import CapabilityContract, Effect
from ueoa.gateway.invokers import InvocationError, Invoker

from .models import (
    DEFAULT_ENTRA_SCOPE,
    AgentCard,
    CheckResult,
    EntraAuth,
    Implementation,
    ImplementationStatus,
    UeoaExtension,
    with_extension,
)
from .ssrf import UrlNotAllowedError, check_credentialed_url, check_url, url_host

WELL_KNOWN_CARD = "/.well-known/agent-card.json"


class RegistryError(ValueError):
    pass


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def _sha256(raw_card: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(raw_card, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _parse_card(raw_card: Any) -> AgentCard:
    try:
        return AgentCard.model_validate(raw_card)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'card'}: {e['msg']}" for e in exc.errors())
        raise RegistryError(f"invalid agent card: {problems}") from exc


def _parse_overlay(overlay: dict[str, Any]) -> UeoaExtension:
    try:
        return UeoaExtension.model_validate(overlay)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'overlay'}: {e['msg']}" for e in exc.errors())
        raise RegistryError(f"invalid overlay: {problems}") from exc


class Registry:
    def __init__(
        self,
        capabilities: dict[str, CapabilityContract],
        invoker: Invoker,
        http: httpx.AsyncClient,
        allowed_hosts: Iterable[str],
        *,
        entra_scopes: Iterable[str] = (DEFAULT_ENTRA_SCOPE,),
    ) -> None:
        self.capabilities = capabilities
        self._invoker = invoker
        self._http = http
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self._entra_scopes = frozenset(entra_scopes)
        self._implementations: dict[str, Implementation] = {}

    # ------------------------------------------------------------- queries

    def implementations(self) -> list[Implementation]:
        return sorted(self._implementations.values(), key=lambda impl: impl.key)

    def get(self, key: str) -> Implementation:
        try:
            return self._implementations[key]
        except KeyError as exc:
            raise RegistryError(f"implementation '{key}' not found") from exc

    def implementations_of(self, capability_id: str) -> list[Implementation]:
        capability = self.capabilities[capability_id]
        tag = f"{capability_id}@{_major(capability.version)}"
        return [impl for impl in self._implementations.values() if tag in impl.card.ueoa.implements]

    def resolve(self, capability_id: str) -> Implementation:
        if capability_id not in self.capabilities:
            raise RegistryError(f"capability '{capability_id}' is not registered")
        candidates = [
            impl for impl in self.implementations_of(capability_id) if impl.status is ImplementationStatus.APPROVED
        ]
        if not candidates:
            raise RegistryError(f"no approved implementation for '{capability_id}'")
        return max(candidates, key=lambda impl: (impl.priority, impl.card.version))

    # ------------------------------------------------------------- registration

    async def register_seed(self, raw_card: dict[str, Any]) -> Implementation:
        """Code-reviewed seed cards (in-process tools) still run conformance, then are approved."""
        card = _parse_card(raw_card)
        impl = Implementation(card=card, source="seed", card_sha256=_sha256(raw_card))
        impl.checks = await self._conformance(card)
        if not impl.conformant:
            failed = [c.check for c in impl.checks if not c.passed]
            raise RegistryError(f"seed '{card.key}' failed conformance: {failed}")
        impl.status = ImplementationStatus.APPROVED
        self._implementations[impl.key] = impl
        return impl

    async def register_from_url(self, url: str, overlay: dict[str, Any] | None = None) -> Implementation:
        """Pull path: fetch the agent's published card. `overlay` supplies the UEOA extension if the card lacks it.

        The overlay may move the card (`a2a.card_path`) and require Entra auth to read it (`auth.scheme: entra`).
        """
        extension = _parse_overlay(overlay) if overlay is not None else None
        card_path = extension.a2a.card_path if extension and extension.a2a else None
        if card_path:
            card_url = f"{url.rstrip('/')}/{card_path.lstrip('/')}"
        else:
            card_url = url if url.endswith(".json") else url.rstrip("/") + WELL_KNOWN_CARD
        try:
            check_url(card_url, self._allowed_hosts)
        except UrlNotAllowedError as exc:
            raise RegistryError(str(exc)) from exc
        headers: dict[str, str] = {}
        if extension and isinstance(extension.auth, EntraAuth):
            self._check_entra_scope(extension.auth)
            try:
                check_credentialed_url(card_url, self._allowed_hosts)
            except UrlNotAllowedError as exc:
                raise RegistryError(str(exc)) from exc
            try:
                headers = await self._invoker.auth_headers(extension.auth, extension.id, card_url)
            except InvocationError as exc:
                raise RegistryError(f"could not authenticate to fetch the agent card: {exc}") from exc
        try:
            response = await self._http.get(card_url, headers=headers, timeout=10)
        except httpx.HTTPError as exc:
            raise RegistryError(f"could not fetch agent card: {exc.__class__.__name__}") from exc
        if response.status_code != 200:
            raise RegistryError(f"agent card fetch returned HTTP {response.status_code}")
        try:
            raw_card = response.json()
        except ValueError as exc:
            raise RegistryError("agent card is not JSON") from exc
        if not isinstance(raw_card, dict):
            raise RegistryError("agent card must be a JSON object")
        sha = _sha256(raw_card)  # pyright: ignore[reportUnknownArgumentType]
        if overlay is not None:
            raw_card = with_extension(raw_card, overlay)  # pyright: ignore[reportUnknownArgumentType]
        return await self._admit(_parse_card(raw_card), source=card_url, sha=sha, pinned_host=url_host(url))

    async def register_card(self, raw_card: dict[str, Any]) -> Implementation:
        """Push path: the owner submits the card (Copilot Studio, Foundry, partner APIs, agents without a card URL)."""
        return await self._admit(_parse_card(raw_card), source="submitted", sha=_sha256(raw_card))

    def register_capability(self, raw_contract: dict[str, Any]) -> CapabilityContract:
        """Owners publish the contract their agents implement; the platform ships with none."""
        try:
            capability = CapabilityContract.model_validate(raw_contract)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'contract'}: {e['msg']}" for e in exc.errors())
            raise RegistryError(f"invalid capability contract: {problems}") from exc
        for name, schema in (("input_schema", capability.input_schema), ("output_schema", capability.output_schema)):
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise RegistryError(f"{name} is not a valid JSON Schema: {exc.message}") from exc
        for index, sample in enumerate(capability.samples):
            try:
                jsonschema.validate(sample, capability.input_schema)
            except jsonschema.ValidationError as exc:
                raise RegistryError(f"samples[{index}] does not match input_schema: {exc.message}") from exc
        if capability.id in self.capabilities:
            raise RegistryError(f"capability '{capability.id}' is already registered")
        self.capabilities[capability.id] = capability
        return capability

    async def _admit(self, card: AgentCard, *, source: str, sha: str, pinned_host: str | None = None) -> Implementation:
        for endpoint in card.endpoints():
            try:
                check_url(endpoint, self._allowed_hosts)
            except UrlNotAllowedError as exc:
                raise RegistryError(f"card endpoint rejected: {exc}") from exc
        if isinstance(card.ueoa.auth, EntraAuth):
            self._check_entra(card, pinned_host)
        if card.ueoa.binding == "local":
            raise RegistryError("only code-reviewed seeds may declare a local binding")
        if card.key in self._implementations:
            raise RegistryError(f"'{card.key}' is already registered; publish a new version")
        impl = Implementation(card=card, source=source, card_sha256=sha)
        impl.checks = await self._conformance(card)
        self._implementations[impl.key] = impl
        return impl

    def _check_entra_scope(self, auth: EntraAuth) -> None:
        if auth.scope not in self._entra_scopes:
            raise RegistryError(f"Entra scope '{auth.scope}' is not allowed (UEOA_ENTRA_SCOPES)")

    def _check_entra(self, card: AgentCard, pinned_host: str | None) -> None:
        """Entra tokens go only to loopback or exactly allow-listed hosts, and only to the host registered from."""
        assert isinstance(card.ueoa.auth, EntraAuth)  # noqa: S101 - caller checked
        self._check_entra_scope(card.ueoa.auth)
        for endpoint in card.endpoints():
            try:
                check_credentialed_url(endpoint, self._allowed_hosts)
            except UrlNotAllowedError as exc:
                raise RegistryError(f"card endpoint rejected: {exc}") from exc
        own = card.agent_endpoints()
        pinned = pinned_host or (url_host(own[0]) if own else None)
        moved = sorted({url_host(url) for url in own} - {pinned})
        if moved:
            raise RegistryError(f"Entra agent endpoints must stay on host '{pinned}', not {moved}")

    def approve(self, key: str, priority: int | None = None) -> Implementation:
        impl = self.get(key)
        if not impl.conformant:
            raise RegistryError(f"'{key}' has failing conformance checks and cannot be approved")
        impl.status = ImplementationStatus.APPROVED
        if priority is not None:
            impl.priority = priority
        return impl

    def set_status(self, key: str, status: ImplementationStatus) -> Implementation:
        impl = self.get(key)
        impl.status = status
        return impl

    # ------------------------------------------------------------- conformance

    async def _conformance(self, card: AgentCard) -> list[CheckResult]:
        checks = [CheckResult(check="card_schema", passed=True, detail="A2A card with the UEOA registry extension")]
        ext = card.ueoa
        known: list[CapabilityContract] = []
        for tag in ext.implements:
            capability_id, _, major = tag.partition("@")
            capability = self.capabilities.get(capability_id)
            if capability is None or _major(capability.version) != major:
                checks.append(CheckResult(check="implements_known_capability", passed=False, detail=tag))
            else:
                known.append(capability)
        if known and len(known) == len(ext.implements):
            checks.append(
                CheckResult(check="implements_known_capability", passed=True, detail=", ".join(ext.implements))
            )
        for capability in known:
            consistent = capability.effects is ext.effects
            checks.append(
                CheckResult(
                    check="effects_consistent",
                    passed=consistent,
                    detail=f"{capability.id}: card={ext.effects}, contract={capability.effects}",
                )
            )
        missing = self._invoker.missing_secrets(card)
        if card.secret_refs():
            detail = f"missing: {', '.join(missing)}" if missing else ", ".join(card.secret_refs())
            checks.append(CheckResult(check="secrets_configured", passed=not missing, detail=detail))
        healthy, health_detail = (False, "skipped: secrets missing") if missing else await self._invoker.health(card)
        checks.append(CheckResult(check="health", passed=healthy, detail=health_detail))
        if healthy and ext.mcp is not None:
            checks.extend(await self._mcp_tool_schema(card, known))
        if healthy:
            for capability in known:
                if capability.effects is Effect.WRITE:
                    # samples would cause real side effects outside the approval path
                    checks.append(
                        CheckResult(
                            check=f"contract_sample[{capability.id}]",
                            passed=True,
                            detail="skipped: write capability (never executed during conformance)",
                        )
                    )
                    continue
                checks.extend(await self._contract_samples(card, capability))
        return checks

    async def _mcp_tool_schema(self, card: AgentCard, capabilities: list[CapabilityContract]) -> list[CheckResult]:
        """The tool must not require arguments the capability input cannot carry."""
        try:
            tool = await self._invoker.mcp_tool(card)
        except InvocationError as exc:
            return [CheckResult(check="mcp_tool_schema", passed=False, detail=f"{exc.code}: {exc}")]
        schema = tool.get("inputSchema")
        required = schema.get("required") if isinstance(schema, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        needed = {str(name) for name in required} if isinstance(required, list) else set()  # pyright: ignore[reportUnknownVariableType]
        mapping = card.ueoa.mapping
        mapped = set(mapping.input) if mapping is not None and mapping.input is not None else None
        source = "the input schema" if mapped is None else "mapping.input"
        results: list[CheckResult] = []
        for capability in capabilities:
            available = mapped if mapped is not None else set(capability.input_schema.get("properties") or {})
            missing = sorted(needed - available)
            detail = f"{capability.id}: " + (
                f"tool requires {missing}, not in {source}" if missing else "tool required fields covered"
            )
            results.append(CheckResult(check="mcp_tool_schema", passed=not missing, detail=detail))
        return results

    async def _contract_samples(self, card: AgentCard, capability: CapabilityContract) -> list[CheckResult]:
        results: list[CheckResult] = []
        for index, sample in enumerate(capability.samples):
            name = f"contract_sample[{capability.id}#{index}]"
            try:
                output = await self._invoker.invoke(card, capability.id, sample)
                jsonschema.validate(output, capability.output_schema)
            except InvocationError as exc:
                results.append(CheckResult(check=name, passed=False, detail=f"{exc.code}: {exc}"))
            except jsonschema.ValidationError as exc:
                results.append(CheckResult(check=name, passed=False, detail=f"output schema: {exc.message}"))
            else:
                results.append(CheckResult(check=name, passed=True, detail="output matches contract schema"))
        return results
