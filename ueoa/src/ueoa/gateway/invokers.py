"""Invokers: how the gateway reaches an implementation, by binding (local, A2A, Copilot Studio, MCP, HTTP).

Tool field mapping (`mapping.input` / `mapping.output`, and `http.query` / `http.body`) uses a tiny path language:
`"$input.a.b[0]"`, `"$result.items[*].name"`, optionally `"... | default('text')"` or `| default(<JSON literal>)`.
Values that are not strings starting with `$` are literals; dicts and lists are mapped recursively. A path that
does not resolve (or resolves to null) takes its default; without one, the key is left out.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ueoa import __version__
from ueoa.jsontext import extract_json_object
from ueoa.registry.models import (
    DEFAULT_ENTRA_SCOPE,
    MAPPING_EXPRESSION_RE,
    URL_PLACEHOLDER_RE,
    A2AOptions,
    AgentCard,
    EntraAuth,
    McpBinding,
    OAuth2Auth,
    SecretAuth,
    ToolMapping,
    parse_mapping_default,
)
from ueoa.registry.ssrf import is_loopback

from .auth import OAuthClientCredentialsProvider, TokenError, TokenProvider

DIRECTLINE_USER = "ueoa-gateway"
ADAPTIVE_CARD = "application/vnd.microsoft.card.adaptive"
# JSON-RPC server error codes the SDK host uses for retryable failures
A2A_ERROR_CODES = {-32001: "unavailable", -32002: "rate_limited", -32003: "timeout"}
PENDING_TASK_STATES = {"submitted", "working"}
MCP_ACCEPT = "application/json, text/event-stream"
MCP_MAX_LIST_PAGES = 20
_PATH_SEGMENT_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+|\*)\]")
_MISSING: Any = object()

LocalHandler = Callable[[dict[str, Any]], dict[str, Any]]


class InvocationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code  # unavailable | rate_limited | timeout | remote_error | bad_response | not_allowed


class _McpSessionExpiredError(InvocationError):
    pass


@dataclass(frozen=True)
class _McpSession:
    id: str | None
    protocol_version: str


def load_local_handler(entrypoint: str, allowed_prefixes: tuple[str, ...]) -> LocalHandler:
    module_name, _, func_name = entrypoint.partition(":")
    if not allowed_prefixes or not module_name.startswith(allowed_prefixes) or not func_name:
        raise InvocationError("not_allowed", f"entrypoint '{entrypoint}' is not in an allow-listed package")
    handler = getattr(importlib.import_module(module_name), func_name)
    if not callable(handler):
        raise InvocationError("not_allowed", f"entrypoint '{entrypoint}' is not callable")
    return handler  # type: ignore[no-any-return]


class Invoker:
    """Dispatches by card binding. The HTTP client is injectable so tests run agents in-process."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        poll_interval_s: float = 1.0,
        settle_s: float = 2.0,
        secrets: Mapping[str, str] | None = None,
        local_prefixes: tuple[str, ...] = (),
        token_provider: TokenProvider | None = None,
        loopback_token_provider: TokenProvider | None = None,
        entra_scopes: Iterable[str] = (DEFAULT_ENTRA_SCOPE,),
        oauth_provider: OAuthClientCredentialsProvider | None = None,
    ) -> None:
        self._http = http
        self._poll_interval_s = poll_interval_s
        # Direct Line: quiet period after the last bot activity before the reply counts as complete
        self._settle_s = settle_s
        self._secrets: Mapping[str, str] = os.environ if secrets is None else secrets
        self._local_prefixes = local_prefixes
        self._token_provider = token_provider
        # dev-only provider (fixed token), never used for a non-loopback host
        self._loopback_token_provider = loopback_token_provider
        self._entra_scopes = frozenset(entra_scopes)
        self._oauth = oauth_provider or OAuthClientCredentialsProvider(http)
        # (MCP url, card key) -> session; keyed by card too so sessions never cross credentials
        self._mcp_sessions: dict[tuple[str, str], _McpSession] = {}

    async def invoke(
        self, card: AgentCard, capability: str, payload: dict[str, Any], invocation_id: str | None = None
    ) -> dict[str, Any]:
        ext = card.ueoa
        if ext.binding == "local":
            handler = load_local_handler(ext.entrypoint or "", self._local_prefixes)
            try:
                return await asyncio.wait_for(asyncio.to_thread(handler, payload), ext.timeout_s)
            except TimeoutError as exc:
                raise InvocationError("timeout", f"local tool exceeded {ext.timeout_s}s") from exc
            except (KeyError, ValueError, TypeError) as exc:
                raise InvocationError("remote_error", f"{card.key} rejected the input: {exc}") from exc
        try:
            async with asyncio.timeout(ext.timeout_s):
                if ext.binding == "a2a":
                    return await self._invoke_a2a(card, capability, payload, invocation_id)
                if ext.binding == "mcp":
                    return await self._invoke_mcp(card, payload)
                if ext.binding == "http":
                    return await self._invoke_http(card, payload)
                return await self._invoke_copilot_studio(card, payload)
        except TimeoutError as exc:
            raise InvocationError("timeout", f"{card.key} did not answer within {ext.timeout_s}s") from exc

    def missing_secrets(self, card: AgentCard) -> list[str]:
        return [ref for ref in card.secret_refs() if not self._secrets.get(ref)]

    async def health(self, card: AgentCard) -> tuple[bool, str]:
        ext = card.ueoa
        if ext.binding == "local":
            return True, "in-process"
        if ext.mcp is not None:
            try:
                async with asyncio.timeout(ext.timeout_s):
                    await self.mcp_tool(card)
            except TimeoutError:
                return False, f"MCP tools/list: no answer within {ext.timeout_s}s"
            except InvocationError as exc:
                return False, f"MCP tools/list: {exc.code}: {exc}"
            return True, f"MCP tool '{ext.mcp.tool}' listed by {ext.mcp.url}"
        if ext.health_url:
            try:
                response = await self._http.get(ext.health_url, timeout=5)
            except httpx.HTTPError as exc:
                return False, f"{ext.health_url}: {exc.__class__.__name__}"
            return response.status_code == 200, f"{ext.health_url}: HTTP {response.status_code}"
        if ext.binding == "http":
            return True, "skipped: http tools have no health endpoint"
        if ext.binding == "copilot-studio":
            try:
                await self._directline_token(card)
            except InvocationError as exc:
                return False, f"Direct Line token: {exc}"
            return True, "Direct Line token acquired"
        return True, "no health_url declared; contract samples verify reachability"

    # ------------------------------------------------------------- A2A (JSON-RPC binding, 1.0 and 0.3)

    async def _invoke_a2a(
        self, card: AgentCard, capability: str, payload: dict[str, Any], invocation_id: str | None
    ) -> dict[str, Any]:
        interface = card.jsonrpc_interface()
        if interface is None:
            raise InvocationError("not_allowed", f"{card.key} declares no JSON-RPC interface")
        url, version = interface
        options = card.ueoa.a2a or A2AOptions()
        metadata = {"ueoa.capability": capability, "ueoa.invocation_id": invocation_id}
        headers = await self.auth_headers(card.ueoa.auth, card.key, url)
        legacy = version.startswith("0.")
        part: dict[str, Any]
        if options.message == "text":
            part = {"text": json.dumps(payload, ensure_ascii=False)}
        else:
            part = {"data": payload} if legacy else {"data": payload, "mediaType": "application/json"}
        if legacy:
            method, get_method = "message/send", "tasks/get"
            part = {"kind": options.message, **part}
            message: dict[str, Any] = {"kind": "message", "role": "user", "parts": [part]}
        else:
            method, get_method = "SendMessage", "GetTask"
            headers["A2A-Version"] = version
            message = {"role": "ROLE_USER", "parts": [part]}
        message |= {"messageId": str(uuid.uuid4()), "metadata": metadata}
        result = await self._rpc(url, card.key, method, {"message": message}, headers)
        while (task_id := _pending_task_id(result, card.key)) is not None:
            await asyncio.sleep(self._poll_interval_s)
            result = await self._rpc(url, card.key, get_method, {"id": task_id}, headers)
        return _a2a_output(result, card.key, options)

    async def _rpc(self, url: str, key: str, method: str, params: dict[str, Any], headers: dict[str, str]) -> Any:
        request = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        body = await self._json("POST", url, key, json=request, headers=headers)
        if "error" in body:
            error = body["error"]
            if isinstance(error, dict):
                code = A2A_ERROR_CODES.get(error.get("code"), "remote_error")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                raise InvocationError(code, str(error.get("message", error)))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            raise InvocationError("remote_error", str(error))
        return body.get("result")

    async def auth_headers(
        self, auth: SecretAuth | EntraAuth | OAuth2Auth | None, key: str, url: str
    ) -> dict[str, str]:
        if auth is None:
            return {}
        if isinstance(auth, OAuth2Auth):
            client_id, client_secret = self._secret(auth.client_id_ref), self._secret(auth.client_secret_ref)
            try:
                token = await self._oauth.token(
                    auth.token_url, client_id, client_secret, scope=auth.scope, audience=auth.audience
                )
            except TokenError as exc:
                raise InvocationError(exc.code, f"{key}: {exc}") from exc
            return {auth.header: f"Bearer {token}"}
        if isinstance(auth, EntraAuth):
            if auth.scope not in self._entra_scopes:
                raise InvocationError("not_allowed", f"{key}: Entra scope '{auth.scope}' is not allowed")
            provider = self._token_provider
            if self._loopback_token_provider is not None and is_loopback(url):
                provider = self._loopback_token_provider
            if provider is None:
                raise InvocationError("not_allowed", f"{key} needs Entra auth but no token provider is configured")
            try:
                token = await provider.token(auth.scope)
            except TokenError as exc:
                raise InvocationError("not_allowed", f"{key}: {exc}") from exc
            return {"Authorization": f"Bearer {token}"}
        value = self._secret(auth.secret_ref)
        return {auth.header: f"Bearer {value}" if auth.scheme == "bearer" else value}

    # ------------------------------------------------------------- Copilot Studio (Direct Line 3.0)

    async def _directline_token(self, card: AgentCard) -> str:
        cfg = card.ueoa.copilot_studio
        assert cfg is not None  # noqa: S101 - guaranteed by UeoaExtension validation
        if cfg.token_endpoint:
            body = await self._json("GET", cfg.token_endpoint, card.key)
        else:
            secret = self._secret(cfg.secret_ref or "")
            url = f"{cfg.directline_url.rstrip('/')}/tokens/generate"
            body = await self._json("POST", url, card.key, headers={"Authorization": f"Bearer {secret}"})
        token = body.get("token")
        if not isinstance(token, str) or not token:
            raise InvocationError("bad_response", f"{card.key}: Direct Line returned no token")
        return token

    async def _invoke_copilot_studio(self, card: AgentCard, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = card.ueoa.copilot_studio
        assert cfg is not None  # noqa: S101 - guaranteed by UeoaExtension validation
        base = cfg.directline_url.rstrip("/")
        token = await self._directline_token(card)
        conversation = await self._json(
            "POST", f"{base}/conversations", card.key, headers={"Authorization": f"Bearer {token}"}
        )
        conversation_id = conversation.get("conversationId")
        if not isinstance(conversation_id, str):
            raise InvocationError("bad_response", f"{card.key}: Direct Line returned no conversationId")
        auth = {"Authorization": f"Bearer {conversation.get('token') or token}"}
        text = str(payload.get(cfg.input_field, "")) if cfg.input_field else json.dumps(payload)
        activity = {
            "type": "message",
            "from": {"id": DIRECTLINE_USER},
            "text": text,
            "locale": "en-US",
            "value": payload,
        }
        activities_url = f"{base}/conversations/{conversation_id}/activities"
        sent = await self._json("POST", activities_url, card.key, json=activity, headers=auth)
        sent_id = sent.get("id")
        replies = await self._directline_replies(activities_url, card.key, auth, sent_id)
        return _directline_output(replies, cfg.output, cfg.text_field, card.key)

    async def _directline_replies(self, url: str, key: str, auth: dict[str, str], sent_id: Any) -> list[dict[str, Any]]:
        """Bot messages answering `sent_id` (by replyToId, or unaddressed after our echoed message). Ends on
        endOfConversation, or once a message arrived and `settle_s` passed without bot activity (typing counts)."""
        loop = asyncio.get_running_loop()
        replies: list[dict[str, Any]] = []
        after_sent = not isinstance(sent_id, str)
        last_activity = loop.time()
        watermark: str | None = None
        while True:
            page = await self._json(
                "GET", url, key, headers=auth, params={"watermark": watermark} if watermark else None
            )
            watermark = page.get("watermark") or watermark
            for activity in page.get("activities") or []:
                if not isinstance(activity, dict):
                    continue
                if activity.get("id") == sent_id:  # pyright: ignore[reportUnknownMemberType]
                    after_sent = True
                    continue
                sender = activity.get("from")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if isinstance(sender, dict) and sender.get("id") == DIRECTLINE_USER:  # pyright: ignore[reportUnknownMemberType]
                    continue
                reply_to = activity.get("replyToId")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if not (reply_to == sent_id if reply_to else after_sent):
                    continue  # a greeting or an answer to another message
                kind = activity.get("type")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if kind == "endOfConversation":
                    return [*replies, activity]  # pyright: ignore[reportUnknownVariableType]
                if kind in {"message", "typing"}:
                    last_activity = loop.time()
                    if kind == "message":
                        replies.append(activity)  # pyright: ignore[reportUnknownArgumentType]
            if replies and loop.time() - last_activity >= self._settle_s:
                return replies
            await asyncio.sleep(self._poll_interval_s)

    # ------------------------------------------------------------- MCP (Streamable HTTP, JSON or SSE responses)

    async def mcp_tool(self, card: AgentCard) -> dict[str, Any]:
        """The card's tool definition from the server's tools/list (follows pagination)."""
        cfg = _mcp_config(card)
        cursor: str | None = None
        for _ in range(MCP_MAX_LIST_PAGES):
            result = await self._mcp_request(card, "tools/list", {"cursor": cursor} if cursor else {})
            tools = result.get("tools") if isinstance(result, dict) else None
            if not isinstance(tools, list):
                raise InvocationError("bad_response", f"{card.key}: tools/list returned no tools")
            for tool in tools:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(tool, dict) and tool.get("name") == cfg.tool:  # pyright: ignore[reportUnknownMemberType]
                    return tool  # pyright: ignore[reportUnknownVariableType]
            cursor = result.get("nextCursor")  # pyright: ignore[reportOptionalMemberAccess]
            if not isinstance(cursor, str) or not cursor:
                break
        raise InvocationError("remote_error", f"{card.key}: MCP server does not list tool '{cfg.tool}'")

    async def _invoke_mcp(self, card: AgentCard, payload: dict[str, Any]) -> dict[str, Any]:
        arguments = _map_input(card.ueoa.mapping, payload)
        result = await self._mcp_request(card, "tools/call", {"name": _mcp_config(card).tool, "arguments": arguments})
        return _map_output(card.ueoa.mapping, payload, _mcp_output(result, card.key), card.key)

    async def _mcp_request(self, card: AgentCard, method: str, params: dict[str, Any]) -> Any:
        """One JSON-RPC request in a cached session; an expired session is re-initialised once."""
        cfg = _mcp_config(card)
        headers = await self.auth_headers(card.ueoa.auth, card.key, cfg.url)
        cache_key = (cfg.url, card.key)
        session = self._mcp_sessions.get(cache_key)
        if session is not None:
            try:
                return await self._mcp_rpc(cfg.url, card.key, method, params, headers, session)
            except _McpSessionExpiredError:
                self._mcp_sessions.pop(cache_key, None)
        session = await self._mcp_initialize(cfg, card.key, headers)
        self._mcp_sessions[cache_key] = session
        try:
            return await self._mcp_rpc(cfg.url, card.key, method, params, headers, session)
        except _McpSessionExpiredError as exc:
            self._mcp_sessions.pop(cache_key, None)
            raise InvocationError("remote_error", f"{card.key}: MCP server rejected a fresh session: {exc}") from exc

    async def _mcp_initialize(self, cfg: McpBinding, key: str, headers: dict[str, str]) -> _McpSession:
        params = {
            "protocolVersion": cfg.protocol_version,
            "capabilities": {},
            "clientInfo": {"name": "ueoa-gateway", "version": __version__},
        }
        request_id = str(uuid.uuid4())
        message = {"jsonrpc": "2.0", "id": request_id, "method": "initialize", "params": params}
        response = await self._mcp_post(cfg.url, key, message, headers)
        result = _mcp_result(_mcp_body(response, request_id, key), key)
        version = result.get("protocolVersion") if isinstance(result, dict) else None
        session = _McpSession(
            response.headers.get("Mcp-Session-Id"), version if isinstance(version, str) else cfg.protocol_version
        )
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        await self._mcp_post(cfg.url, key, notification, headers | _mcp_headers(session))
        return session

    async def _mcp_rpc(
        self, url: str, key: str, method: str, params: dict[str, Any], headers: dict[str, str], session: _McpSession
    ) -> Any:
        request_id = str(uuid.uuid4())
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = await self._mcp_post(url, key, message, headers | _mcp_headers(session))
        return _mcp_result(_mcp_body(response, request_id, key), key)

    async def _mcp_post(self, url: str, key: str, message: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        try:
            response = await self._http.post(url, json=message, headers={"Accept": MCP_ACCEPT} | headers)
        except httpx.TimeoutException as exc:
            raise InvocationError("timeout", f"{key} timed out") from exc
        except httpx.TransportError as exc:
            raise InvocationError("unavailable", f"{key} is unreachable: {exc.__class__.__name__}") from exc
        status = response.status_code
        if "Mcp-Session-Id" in headers and (status == 404 or (status == 400 and "session" in response.text.lower())):
            raise _McpSessionExpiredError("remote_error", f"{key} returned HTTP {status} for the MCP session")
        if status == 429:
            raise InvocationError("rate_limited", f"{key} returned HTTP 429")
        if status >= 500:
            raise InvocationError("unavailable", f"{key} returned HTTP {status}")
        if status >= 400:
            raise InvocationError("remote_error", f"{key} returned HTTP {status}")
        return response

    # ------------------------------------------------------------- plain HTTP (REST) tools

    async def _invoke_http(self, card: AgentCard, payload: dict[str, Any]) -> dict[str, Any]:
        ext = card.ueoa
        cfg = ext.http
        assert cfg is not None  # noqa: S101 - guaranteed by UeoaExtension validation
        args = _map_input(ext.mapping, payload)
        scope = {"input": args}
        url = _fill_url(cfg.url, args, card.key)
        params = {k: v for k, v in _render(cfg.query, scope).items() if v is not None} if cfg.query else None
        body = _render(cfg.body, scope) if cfg.body is not None else (None if cfg.method == "GET" else args)
        headers = (
            {"Accept": "application/json"} | (cfg.headers or {}) | await self.auth_headers(ext.auth, card.key, url)
        )
        response = await self._send(cfg.method, url, card.key, params=params, json=body, headers=headers)
        if response.status_code not in cfg.success_status:
            raise InvocationError("remote_error", f"{card.key} returned HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            raise InvocationError("bad_response", f"{card.key} returned non-JSON") from exc
        return _map_output(ext.mapping, payload, result, card.key)

    # ------------------------------------------------------------- shared

    def _secret(self, ref: str) -> str:
        value = self._secrets.get(ref)
        if not value:
            raise InvocationError("not_allowed", f"secret '{ref}' is not configured")
        return value

    async def _send(self, method: str, url: str, key: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise InvocationError("timeout", f"{key} timed out") from exc
        except httpx.TransportError as exc:
            raise InvocationError("unavailable", f"{key} is unreachable: {exc.__class__.__name__}") from exc
        if response.status_code == 429:
            raise InvocationError("rate_limited", f"{key} returned HTTP 429")
        if response.status_code >= 500:
            raise InvocationError("unavailable", f"{key} returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise InvocationError("remote_error", f"{key} returned HTTP {response.status_code}")
        return response

    async def _json(self, method: str, url: str, key: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._send(method, url, key, **kwargs)
        try:
            body = response.json()
        except ValueError as exc:
            raise InvocationError("bad_response", f"{key} returned non-JSON") from exc
        if not isinstance(body, dict):
            raise InvocationError("bad_response", f"{key} returned a non-object body")
        return body  # pyright: ignore[reportUnknownVariableType]


def _text_of(parts: list[Any]) -> str:
    return "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))


def _split_result(result: Any, key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(task, message) from 1.0 ({task}|{message}, or a bare Task from GetTask) and 0.3 (`kind`) results."""
    if not isinstance(result, dict):
        raise InvocationError("bad_response", f"{key} returned no result")
    task, message = result.get("task"), result.get("message")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if task is None and message is None:
        task, message = (result, None) if result.get("kind") == "task" or "status" in result else (None, result)  # pyright: ignore[reportUnknownMemberType]
    return (task if isinstance(task, dict) else None), (message if isinstance(message, dict) else None)  # pyright: ignore[reportUnknownVariableType]


def _task_state(task: dict[str, Any]) -> str:
    return str((task.get("status") or {}).get("state", "")).lower().removeprefix("task_state_").replace("_", "-")


def _pending_task_id(result: Any, key: str) -> str | None:
    task, _ = _split_result(result, key)
    if task is None or _task_state(task) not in PENDING_TASK_STATES:
        return None
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise InvocationError("bad_response", f"{key} returned an unfinished task without an id")
    return task_id


def _a2a_output(result: Any, key: str, options: A2AOptions) -> dict[str, Any]:
    """Prefer a structured data part; otherwise parse the text parts (or wrap them in output mode `text`)."""
    task, message = _split_result(result, key)
    if task is not None:
        status = task.get("status") or {}
        state = _task_state(task)
        status_text = _text_of((status.get("message") or {}).get("parts") or [])
        if state not in {"", "completed"}:
            raise InvocationError("remote_error", f"{key} task ended in state '{state}': {status_text}".rstrip(": "))
        parts = [p for artifact in task.get("artifacts") or [] for p in artifact.get("parts") or []]
        parts = parts or (status.get("message") or {}).get("parts") or []
    else:
        parts = (message or {}).get("parts") or []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            return part["data"]  # pyright: ignore[reportUnknownVariableType]
    text = _text_of(parts).strip()
    if options.output == "text":
        if not text:
            raise InvocationError("bad_response", f"{key} returned no text")
        try:
            parsed: Any = json.loads(text)
        except ValueError:
            parsed = None
        return parsed if isinstance(parsed, dict) else {options.text_field: text}  # pyright: ignore[reportUnknownVariableType]
    parsed = extract_json_object(text)
    if parsed is None:
        raise InvocationError("bad_response", f"{key} returned no JSON output")
    return parsed


def _directline_output(replies: list[dict[str, Any]], mode: str, text_field: str, key: str) -> dict[str, Any]:
    for reply in replies:
        if isinstance(reply.get("value"), dict):
            return reply["value"]
    text = "\n".join(part for reply in replies if (part := _activity_text(reply))).strip()
    if not text:
        raise InvocationError("bad_response", f"{key} sent no reply")
    if mode == "text":
        return {text_field: text}
    parsed = extract_json_object(text)
    if parsed is None:
        raise InvocationError("bad_response", f"{key} reply is not JSON")
    return parsed


def _activity_text(activity: dict[str, Any]) -> str:
    """Message text plus the TextBlock / TextRun / FactSet text of Adaptive Card attachments."""
    text = activity.get("text")
    parts = [text] if isinstance(text, str) and text.strip() else []
    for attachment in activity.get("attachments") or []:
        if isinstance(attachment, dict) and attachment.get("contentType") == ADAPTIVE_CARD:  # pyright: ignore[reportUnknownMemberType]
            _card_text(attachment.get("content"), parts)  # pyright: ignore[reportUnknownMemberType]
    return "\n".join(parts)


def _card_text(node: Any, out: list[str]) -> None:
    if isinstance(node, list):
        for item in node:  # pyright: ignore[reportUnknownVariableType]
            _card_text(item, out)
        return
    if not isinstance(node, dict):
        return
    kind, text = node.get("type"), node.get("text")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if kind in {"TextBlock", "TextRun"} and isinstance(text, str):
        out.append(text)
    elif kind == "FactSet":
        for fact in node.get("facts") or []:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(fact, dict):
                out.append(f"{fact.get('title', '')}: {fact.get('value', '')}")  # pyright: ignore[reportUnknownMemberType]
    for name, child in node.items():  # pyright: ignore[reportUnknownVariableType]
        if name not in {"actions", "selectAction", "facts"} and isinstance(child, (dict, list)):
            _card_text(child, out)


# ------------------------------------------------------------- tool field mapping


def _walk(value: Any, segments: list[tuple[str, str]]) -> Any:
    for position, (name, index) in enumerate(segments):
        if name:
            if not isinstance(value, dict) or name not in value:
                return _MISSING
            value = value[name]
        elif index == "*":
            if not isinstance(value, list):
                return _MISSING
            rest = segments[position + 1 :]
            return [v for v in (_walk(item, rest) for item in value) if v is not _MISSING]  # pyright: ignore[reportUnknownVariableType]
        else:
            if not isinstance(value, list) or int(index) >= len(value):  # pyright: ignore[reportUnknownArgumentType]
                return _MISSING
            value = value[int(index)]
    return value


def _render(template: Any, scope: dict[str, Any]) -> Any:
    """Resolve a mapping template against `scope` (root name -> value); see the module docstring."""
    if isinstance(template, dict):
        rendered = {k: _render(v, scope) for k, v in template.items()}  # pyright: ignore[reportUnknownVariableType]
        return {k: v for k, v in rendered.items() if v is not _MISSING}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(template, list):
        return [v for v in (_render(item, scope) for item in template) if v is not _MISSING]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(template, str) and template.lstrip().startswith("$"):
        match = MAPPING_EXPRESSION_RE.match(template.strip())
        if match is None:
            return _MISSING
        value = _walk(scope.get(match["root"], _MISSING), _PATH_SEGMENT_RE.findall(match["path"]))
        if (value is _MISSING or value is None) and match["default"] is not None:
            return parse_mapping_default(match["default"])
        return value
    return template


def _map_input(mapping: ToolMapping | None, payload: dict[str, Any]) -> dict[str, Any]:
    if mapping is None or mapping.input is None:
        return payload
    return _render(mapping.input, {"input": payload})


def _map_output(mapping: ToolMapping | None, payload: dict[str, Any], result: Any, key: str) -> dict[str, Any]:
    if mapping is not None and mapping.output is not None:
        return _render(mapping.output, {"input": payload, "result": result})
    if not isinstance(result, dict):
        raise InvocationError("bad_response", f"{key} returned a non-object body")
    return result  # pyright: ignore[reportUnknownVariableType]


def _fill_url(template: str, values: dict[str, Any], key: str) -> str:
    def fill(match: re.Match[str]) -> str:
        value = values.get(match[1])
        if value is None or isinstance(value, (dict, list)):
            raise InvocationError("remote_error", f"{key}: no scalar value for URL placeholder '{match[1]}'")
        return quote(json.dumps(value) if isinstance(value, bool) else str(value), safe="")

    return URL_PLACEHOLDER_RE.sub(fill, template)


def _mcp_config(card: AgentCard) -> McpBinding:
    cfg = card.ueoa.mcp
    if cfg is None:
        raise InvocationError("not_allowed", f"{card.key} has no MCP binding")
    return cfg


def _mcp_headers(session: _McpSession) -> dict[str, str]:
    headers = {"MCP-Protocol-Version": session.protocol_version}
    if session.id:
        headers["Mcp-Session-Id"] = session.id
    return headers


def _mcp_body(response: httpx.Response, request_id: str, key: str) -> dict[str, Any]:
    """The JSON-RPC response for `request_id`, from a JSON body or from the `data:` lines of an SSE stream."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for event in response.text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].removeprefix(" ") for line in event.split("\n") if line.startswith("data:")
            ).strip()
            if not data:
                continue
            try:
                message = json.loads(data)
            except ValueError:
                continue
            if isinstance(message, dict) and message.get("id") == request_id:  # pyright: ignore[reportUnknownMemberType]
                return message  # pyright: ignore[reportUnknownVariableType]
        raise InvocationError("bad_response", f"{key}: SSE stream carried no response to the request")
    try:
        body = response.json()
    except ValueError as exc:
        raise InvocationError("bad_response", f"{key} returned non-JSON") from exc
    if not isinstance(body, dict) or body.get("id") != request_id:  # pyright: ignore[reportUnknownMemberType]
        raise InvocationError("bad_response", f"{key} returned no JSON-RPC response to the request")
    return body  # pyright: ignore[reportUnknownVariableType]


def _mcp_result(body: dict[str, Any], key: str) -> Any:
    if "error" in body:
        error = body["error"]
        message = error.get("message", error) if isinstance(error, dict) else error  # pyright: ignore[reportUnknownMemberType]
        raise InvocationError("remote_error", f"{key}: {message}")
    return body.get("result")


def _mcp_output(result: Any, key: str) -> dict[str, Any]:
    """structuredContent if it is an object, else the first text content as JSON, else {"text": ...}."""
    if not isinstance(result, dict):
        raise InvocationError("bad_response", f"{key}: tools/call returned no result")
    content = result.get("content")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    texts = [
        c["text"]
        for c in (content if isinstance(content, list) else [])  # pyright: ignore[reportUnknownVariableType]
        if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str)  # pyright: ignore[reportUnknownMemberType]
    ]
    if result.get("isError") is True:  # pyright: ignore[reportUnknownMemberType]
        raise InvocationError("remote_error", f"{key}: {' '.join(texts) or 'tool reported an error'}")
    structured = result.get("structuredContent")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(structured, dict):
        return structured  # pyright: ignore[reportUnknownVariableType]
    if not texts:
        raise InvocationError("bad_response", f"{key}: tool returned no structured or text content")
    parsed = extract_json_object(texts[0])
    return parsed if parsed is not None else {"text": texts[0]}
