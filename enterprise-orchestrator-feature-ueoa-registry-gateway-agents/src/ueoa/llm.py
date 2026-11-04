"""LLM client: EY foundation-layer gateway, secure route, JSON-only answers."""

from __future__ import annotations

import asyncio
import os
import ssl
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import truststore

from ueoa.jsontext import extract_json_object

_REQUIRED = ("UEOA_LLM_URL", "UEOA_LLM_MODEL", "UEOA_LLM_AGENT_ID", "UEOA_LLM_PROJECT_ID")


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    """Transient: 5xx, transport failure, timeout or the overall deadline. Safe to retry."""


class LLMRateLimitedError(LLMError):
    """HTTP 429 from the gateway. Safe to retry after a back-off."""


class JsonLLM(Protocol):
    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FoundationSettings:
    # no temperature/response_format: the gateway payload (its OpenAPI) has no such fields; JSON is prompt-enforced
    url: str  # .../api/agentic_openai/generate_response
    model: str
    agent_id: str
    project_id: str
    project_name: str
    environment: str
    key: str  # "foundation key": password for /get_token; empty -> unauthenticated route

    @classmethod
    def from_env(cls) -> FoundationSettings:
        missing = [name for name in _REQUIRED if not os.environ.get(name)]
        if missing:
            raise LLMError(f"LLM settings missing: {', '.join(missing)} (see .env.example)")
        env = os.environ
        return cls(
            url=env["UEOA_LLM_URL"],
            model=env["UEOA_LLM_MODEL"],
            agent_id=env["UEOA_LLM_AGENT_ID"],
            project_id=env["UEOA_LLM_PROJECT_ID"],
            project_name=env.get("UEOA_LLM_PROJECT_NAME", env["UEOA_LLM_PROJECT_ID"]),
            environment=env.get("UEOA_LLM_ENVIRONMENT", "DEV"),
            key=env.get("UEOA_LLM_FOUNDATION_KEY", ""),
        )


class FoundationLLM:
    def __init__(
        self,
        settings: FoundationSettings,
        *,
        agent_name: str,
        http: httpx.AsyncClient | None = None,
        deadline_s: float = 110,
    ):
        self._settings = settings
        self._agent_name = agent_name
        self._deadline_s = deadline_s
        self._http = http or httpx.AsyncClient(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=120)
        self._token: str | None = None
        self._lock = asyncio.Lock()
        base, sep, _ = settings.url.partition("/api/")
        if not sep:
            raise LLMError("UEOA_LLM_URL must point at .../api/agentic_openai/generate_response")
        self._token_url = f"{base}/get_token"
        self._route = (
            settings.url.replace("generate_response", "generate_secure_response") if settings.key else settings.url
        )

    async def _headers(self, *, refresh: bool) -> dict[str, str]:
        if not self._settings.key:
            return {}
        async with self._lock:
            if self._token is None or refresh:
                response = await self._post(
                    self._token_url, "foundation get_token", json={"password": self._settings.key}
                )
                if response.status_code != 200:
                    raise _status_error("foundation get_token", response.status_code)
                self._token = response.json().get("access_token")
                if not self._token:
                    raise LLMError("foundation get_token returned no access_token")
        return {"Authorization": f"Bearer {self._token}"}

    async def _post(self, url: str, what: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._http.post(url, **kwargs)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(f"{what} timed out") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(f"{what} unreachable: {exc.__class__.__name__}") from exc

    async def complete(self, system: str, user: str, *, chain_id: str | None = None) -> str:
        s = self._settings
        payload = {
            "model_name": s.model,
            "system_prompt": system,
            "user_prompt": user,
            "agent_id": s.agent_id,
            "metadata": {"user_id": "ueoa", "chain_id": chain_id, "parent_id": self._agent_name},
            "project_context": {
                "project_id": s.project_id,
                "project_name": s.project_name,
                "environment": s.environment,
            },
        }
        what = "LLM gateway"
        response = await self._post(self._route, what, json=payload, headers=await self._headers(refresh=False))
        if response.status_code == 401 and s.key:
            response = await self._post(self._route, what, json=payload, headers=await self._headers(refresh=True))
        if response.status_code != 200:
            raise _status_error(what, response.status_code, _gateway_message(response))
        try:
            return str(response.json()["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("unexpected LLM response shape") from exc

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self._deadline_s):
                parsed = extract_json_object(await self.complete(system, user, chain_id=chain_id))
                if parsed is None:
                    retry = f"{user}\n\nYour previous reply was not valid JSON. Reply with exactly one JSON object."
                    parsed = extract_json_object(await self.complete(system, retry, chain_id=chain_id))
        except TimeoutError as exc:
            raise LLMUnavailableError("deadline") from exc
        if parsed is None:
            raise LLMError("model did not return a JSON object")
        return parsed


def _gateway_message(response: httpx.Response) -> str:
    """The gateway's own short rejection reason (e.g. its content filter), never echoing the prompt."""
    try:
        message = response.json().get("message")
    except (ValueError, AttributeError):
        return ""
    return str(message)[:200] if isinstance(message, str) else ""


def _status_error(what: str, status: int, detail: str = "") -> LLMError:
    message = f"{what} returned HTTP {status}" + (f": {detail}" if detail else "")
    if status == 429:
        return LLMRateLimitedError(message)
    if status >= 500:
        return LLMUnavailableError(message)
    return LLMError(message)
