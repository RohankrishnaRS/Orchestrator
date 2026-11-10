"""Agent cards (A2A AgentCard, 0.3 and 1.0 shapes) carrying the UEOA registry extension, and registry records."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ueoa.contracts.common import SEMVER_PATTERN, Effect

UEOA_EXTENSION_URI = "https://ueoa.ey.com/a2a/extensions/registry/v1"
# secrets are referenced by environment-variable name only, and only from this namespace
SECRET_REF_PATTERN = r"^UEOA_SECRET_[A-Z0-9_]+$"  # noqa: S105 - a name pattern, not a secret
DEFAULT_ENTRA_SCOPE = "https://ai.azure.com/.default"
# Tool field mapping: "$<root>.a.b[0].c[*].d" optionally followed by "| default(<JSON literal or 'text'>)"
MAPPING_EXPRESSION_RE = re.compile(
    r"^\$(?P<root>[a-z]+)(?P<path>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\]|\[\*\])*)"
    r"(?:\s*\|\s*default\((?P<default>.*)\))?$",
    re.DOTALL,
)
URL_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# header names that would carry credentials; those go through `auth` only
SECRET_HEADER_HINTS = ("authorization", "cookie", "key", "token", "secret", "password", "auth")


def parse_mapping_default(text: str) -> Any:
    """The fallback of `| default(...)`: a single-quoted string or a JSON literal."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f"default({text}) is not a quoted string or JSON literal") from exc


def _check_mapping(value: Any, roots: frozenset[str], where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            _check_mapping(item, roots, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            _check_mapping(item, roots, f"{where}[{index}]")
    elif isinstance(value, str) and value.lstrip().startswith("$"):
        match = MAPPING_EXPRESSION_RE.match(value.strip())
        if match is None:
            raise ValueError(f"{where}: invalid expression '{value}'")
        if match["root"] not in roots:
            raise ValueError(f"{where}: unknown root ${match['root']}; allowed: {sorted(roots)}")
        if match["default"] is not None:
            parse_mapping_default(match["default"])


class AgentSkill(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class SecretAuth(BaseModel):
    """How the gateway authenticates to an agent. Only a reference to the secret is stored."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["bearer", "api-key"] = "bearer"
    header: str = "Authorization"
    secret_ref: str = Field(pattern=SECRET_REF_PATTERN)


class EntraAuth(BaseModel):
    """Microsoft Entra ID bearer token acquired by the platform identity for `scope`. No secret is stored."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["entra"]
    scope: str = DEFAULT_ENTRA_SCOPE


class OAuth2Auth(BaseModel):
    """OAuth 2.0 client-credentials grant; client id and secret are referenced by name only."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["oauth2"]
    token_url: str
    client_id_ref: str = Field(pattern=SECRET_REF_PATTERN)
    client_secret_ref: str = Field(pattern=SECRET_REF_PATTERN)
    scope: str | None = None
    audience: str | None = None
    header: str = "Authorization"


class A2AOptions(BaseModel):
    """How the gateway talks to an A2A agent: payload as a data part or JSON text, output as JSON or plain text."""

    model_config = ConfigDict(extra="forbid")

    message: Literal["data", "text"] = "data"
    output: Literal["json", "text"] = "json"
    text_field: str = "answer"
    card_path: str | None = None  # agent card path relative to the agent URL, instead of /.well-known/agent-card.json


class CopilotStudioBinding(BaseModel):
    """Copilot Studio agent reached over Direct Line 3.0 (token endpoint or Direct Line secret)."""

    model_config = ConfigDict(extra="forbid")

    directline_url: str = "https://directline.botframework.com/v3/directline"
    token_endpoint: str | None = None
    secret_ref: str | None = Field(default=None, pattern=SECRET_REF_PATTERN)
    output: Literal["json", "text"] = "json"
    text_field: str = "answer"
    input_field: str | None = None  # send this input field as the message text instead of the whole input as JSON

    @model_validator(mode="after")
    def _one_credential(self) -> CopilotStudioBinding:
        if not self.token_endpoint and not self.secret_ref:
            raise ValueError("copilot_studio needs a token_endpoint or a secret_ref")
        return self


class McpBinding(BaseModel):
    """One tool on a remote MCP server reached over Streamable HTTP."""

    model_config = ConfigDict(extra="forbid")

    url: str  # the MCP endpoint
    tool: str = Field(min_length=1)
    protocol_version: str = "2025-06-18"


class HttpBinding(BaseModel):
    """A plain REST endpoint. `{name}` placeholders in the path, `query` and `body` are filled from `$input`."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["GET", "POST", "PUT", "PATCH"] = "GET"
    url: str
    query: dict[str, Any] | None = None
    headers: dict[str, str] | None = None  # static and non-secret; credentials only via `auth`
    body: dict[str, Any] | None = None
    success_status: list[int] = Field(default_factory=lambda: [200, 201], min_length=1)

    @model_validator(mode="after")
    def _check(self) -> HttpBinding:
        origin, sep, _ = self.url.partition("://")
        authority = self.url[len(origin) + len(sep) :].split("/", 1)[0]
        if not sep or "{" in authority or "}" in authority:
            raise ValueError("http.url must be absolute, with placeholders only in the path")
        for name in self.headers or {}:
            if any(hint in name.lower() for hint in SECRET_HEADER_HINTS):
                raise ValueError(f"http.headers: '{name}' looks like a credential; use auth instead")
        if any(not 200 <= status < 300 for status in self.success_status):
            raise ValueError("http.success_status must be 2xx codes")
        _check_mapping(self.query, frozenset({"input"}), "http.query")
        _check_mapping(self.body, frozenset({"input"}), "http.body")
        return self


class ToolMapping(BaseModel):
    """Field mapping between the capability and a tool: `input` from `$input`, `output` from `$result`/`$input`."""

    model_config = ConfigDict(extra="forbid")

    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> ToolMapping:
        _check_mapping(self.input, frozenset({"input"}), "mapping.input")
        _check_mapping(self.output, frozenset({"input", "result"}), "mapping.output")
        return self


class UeoaExtension(BaseModel):
    """Registry metadata carried in the card under capabilities.extensions[uri=UEOA_EXTENSION_URI].params."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    owner: str
    kind: Literal["agent", "tool", "workflow"]
    implementation: Literal["deterministic", "rule_based", "llm_function", "llm_agent", "low_code", "workflow"]
    binding: Literal["local", "a2a", "copilot-studio", "mcp", "http"]
    implements: list[str] = Field(min_length=1)  # "capability.id@major"
    effects: Effect
    timeout_s: int = Field(gt=0, le=600)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    data_classification_ceiling: Literal["public", "internal", "confidential"] = "internal"
    health_url: str | None = None
    entrypoint: str | None = None  # local binding only: "module:function" under an allow-listed package
    auth: SecretAuth | EntraAuth | OAuth2Auth | None = None
    a2a: A2AOptions | None = None
    copilot_studio: CopilotStudioBinding | None = None
    mcp: McpBinding | None = None
    http: HttpBinding | None = None
    mapping: ToolMapping | None = None

    @model_validator(mode="after")
    def _binding_config(self) -> UeoaExtension:
        if (self.binding == "local") != (self.entrypoint is not None):
            raise ValueError("entrypoint is required for, and only allowed with, binding 'local'")
        if (self.binding == "copilot-studio") != (self.copilot_studio is not None):
            raise ValueError(
                "copilot_studio settings are required for, and only allowed with, binding 'copilot-studio'"
            )
        if (self.binding == "mcp") != (self.mcp is not None):
            raise ValueError("mcp settings are required for, and only allowed with, binding 'mcp'")
        if (self.binding == "http") != (self.http is not None):
            raise ValueError("http settings are required for, and only allowed with, binding 'http'")
        if self.a2a is not None and self.binding != "a2a":
            raise ValueError("a2a settings are only allowed with binding 'a2a'")
        if self.mapping is not None and self.binding not in {"mcp", "http"}:
            raise ValueError("mapping is only allowed with bindings 'mcp' and 'http'")
        if self.http is not None and self.http.method == "GET" and self.effects is not Effect.READ:
            raise ValueError("http method GET is only allowed with effects 'read'")
        return self


class AgentExtension(BaseModel):
    model_config = ConfigDict(extra="allow")

    uri: str
    description: str | None = None
    required: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class AgentCapabilities(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    streaming: bool | None = None
    push_notifications: bool | None = Field(default=None, alias="pushNotifications")
    extensions: list[AgentExtension] = Field(default_factory=lambda: [])


class AgentInterface(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    url: str
    protocol_binding: str = Field(default="JSONRPC", alias="protocolBinding")
    protocol_version: str = Field(default="1.0", alias="protocolVersion")


class AgentCard(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    description: str
    version: str = Field(pattern=SEMVER_PATTERN)
    url: str | None = None  # A2A 0.3 endpoint
    protocol_version: str | None = Field(default=None, alias="protocolVersion")  # A2A 0.3
    supported_interfaces: list[AgentInterface] = Field(default_factory=lambda: [], alias="supportedInterfaces")
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(min_length=1)
    default_input_modes: list[str] = Field(default_factory=lambda: ["application/json"], alias="defaultInputModes")
    default_output_modes: list[str] = Field(default_factory=lambda: ["application/json"], alias="defaultOutputModes")

    _ueoa: UeoaExtension = PrivateAttr()

    @model_validator(mode="after")
    def _extract_extension(self) -> AgentCard:
        matches = [ext for ext in self.capabilities.extensions if ext.uri == UEOA_EXTENSION_URI]
        if len(matches) != 1:
            raise ValueError(
                f"card must declare exactly one capabilities.extensions entry with uri '{UEOA_EXTENSION_URI}' "
                "(or be registered with an overlay)"
            )
        self._ueoa = UeoaExtension.model_validate(matches[0].params)
        return self

    @property
    def ueoa(self) -> UeoaExtension:
        return self._ueoa

    @property
    def key(self) -> str:
        return f"{self._ueoa.id}@{self.version}"

    def jsonrpc_interface(self) -> tuple[str, str] | None:
        """(url, protocol version) of the preferred JSON-RPC interface: 1.0 supportedInterfaces, else 0.3 url."""
        for interface in self.supported_interfaces:
            if interface.protocol_binding.upper() == "JSONRPC":
                return interface.url, interface.protocol_version
        if self.url:
            return self.url, self.protocol_version or "0.3"
        return None

    def endpoints(self) -> list[str]:
        """Every URL the platform would call for this card (checked against the allow-list at registration)."""
        ext = self._ueoa
        urls = [interface.url for interface in self.supported_interfaces]
        urls += [u for u in (self.url, ext.health_url) if u]
        if ext.copilot_studio:
            urls += [u for u in (ext.copilot_studio.directline_url, ext.copilot_studio.token_endpoint) if u]
        if ext.mcp:
            urls.append(ext.mcp.url)
        if ext.http:
            urls.append(ext.http.url)
        if isinstance(ext.auth, OAuth2Auth):
            urls.append(ext.auth.token_url)
        return urls

    def agent_endpoints(self) -> list[str]:
        """The agent's own URLs (MCP/HTTP endpoint, interfaces, 0.3 url, health); an Entra agent keeps one host."""
        urls = [self._ueoa.mcp.url] if self._ueoa.mcp else []
        urls += [self._ueoa.http.url] if self._ueoa.http else []
        urls += [interface.url for interface in self.supported_interfaces]
        return urls + [u for u in (self.url, self._ueoa.health_url) if u]

    def secret_refs(self) -> list[str]:
        ext = self._ueoa
        refs = [ext.auth.secret_ref] if isinstance(ext.auth, SecretAuth) else []
        if isinstance(ext.auth, OAuth2Auth):
            refs += [ext.auth.client_id_ref, ext.auth.client_secret_ref]
        if ext.copilot_studio and ext.copilot_studio.secret_ref:
            refs.append(ext.copilot_studio.secret_ref)
        return refs


def with_extension(raw_card: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Attach (or replace) the UEOA extension on a card that was published without it."""
    card = copy.deepcopy(raw_card)
    capabilities = card.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    extensions = [
        ext
        for ext in capabilities.get("extensions") or []  # pyright: ignore[reportUnknownVariableType]
        if not (isinstance(ext, dict) and ext.get("uri") == UEOA_EXTENSION_URI)  # pyright: ignore[reportUnknownMemberType]
    ]
    extensions.append({"uri": UEOA_EXTENSION_URI, "params": params})
    card["capabilities"] = {**capabilities, "extensions": extensions}
    return card


class ImplementationStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


class CheckResult(BaseModel):
    check: str
    passed: bool
    detail: str = ""


class Implementation(BaseModel):
    card: AgentCard
    status: ImplementationStatus = ImplementationStatus.DRAFT
    priority: int = 10
    source: str  # "seed", "submitted" or the card URL
    card_sha256: str = ""
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checks: list[CheckResult] = Field(default_factory=lambda: [])

    @property
    def key(self) -> str:
        return self.card.key

    @property
    def conformant(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)
