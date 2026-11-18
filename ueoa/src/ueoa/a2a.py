"""A2A server side (JSON-RPC binding): request parsing, skill dispatch, task replies and error codes.

Accepts A2A 1.0 (`SendMessage`) and 0.3 (`message/send`) requests and answers in the matching shape.
Input is the first structured `data` part (or a JSON text part); output is returned as a task artifact.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ueoa.jsontext import extract_json_object
from ueoa.llm import LLMError, LLMRateLimitedError, LLMUnavailableError

METHODS = frozenset({"SendMessage", "message/send"})


@dataclass(frozen=True)
class InvocationContext:
    capability: str
    invocation_id: str | None


SkillHandler = Callable[[dict[str, Any], InvocationContext], Awaitable[dict[str, Any]]]


async def handle_jsonrpc(body: Any, handlers: Mapping[str, SkillHandler]) -> dict[str, Any]:
    """JSON-RPC response for one request. `body` is the parsed request, or the raw bytes / text to parse."""
    if isinstance(body, bytes | bytearray | str):
        try:
            body = json.loads(body)
        except ValueError:
            return error_response(None, -32700, "Invalid JSON payload")
    request: dict[str, Any] = body if isinstance(body, dict) else {}  # pyright: ignore[reportUnknownVariableType]
    request_id = request.get("id")
    method = request.get("method")
    if method not in METHODS:
        return error_response(request_id, -32601, "Method not found")
    params = request.get("params") or {}
    message = params.get("message") if isinstance(params, dict) else None
    if not isinstance(message, dict):
        return error_response(request_id, -32602, "params.message is required")
    metadata = {**_dict(params.get("metadata")), **_dict(message.get("metadata"))}  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    capability = metadata.get("ueoa.capability") or (next(iter(handlers)) if len(handlers) == 1 else None)
    handler = handlers.get(str(capability))
    if handler is None:
        return error_response(request_id, -32602, f"unknown capability {capability!r}")
    parts = message.get("parts")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    payload = input_from_parts(parts if isinstance(parts, list) else [])  # pyright: ignore[reportUnknownArgumentType]
    if payload is None:
        return error_response(request_id, -32602, "message needs a data part or a JSON text part")
    try:
        output = await handler(payload, InvocationContext(str(capability), metadata.get("ueoa.invocation_id")))
    except (KeyError, ValueError, TypeError) as exc:
        return error_response(request_id, -32602, f"invalid input: {exc}")
    except LLMRateLimitedError as exc:
        return error_response(request_id, -32002, f"model rate limited: {exc}")
    except LLMUnavailableError as exc:
        return error_response(request_id, -32001, f"model unavailable: {exc}")
    except TimeoutError:
        return error_response(request_id, -32003, "skill timed out")
    except LLMError as exc:
        return error_response(request_id, -32603, f"model call failed: {exc}")
    return {"jsonrpc": "2.0", "id": request_id, "result": task_result(output, legacy=method != "SendMessage")}


def input_from_parts(parts: list[Any]) -> dict[str, Any] | None:
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            return part["data"]  # pyright: ignore[reportUnknownVariableType]
    text = "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
    return extract_json_object(text) if text else None


def task_result(output: dict[str, Any], *, legacy: bool) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    ids = {"id": str(uuid.uuid4()), "contextId": str(uuid.uuid4())}
    artifact: dict[str, Any] = {"artifactId": str(uuid.uuid4()), "name": "output"}
    if legacy:
        artifact["parts"] = [{"kind": "data", "data": output}]
        return {"kind": "task", **ids, "status": {"state": "completed", "timestamp": now}, "artifacts": [artifact]}
    artifact["parts"] = [{"data": output, "mediaType": "application/json"}]
    return {"task": {**ids, "status": {"state": "TASK_STATE_COMPLETED", "timestamp": now}, "artifacts": [artifact]}}


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]
