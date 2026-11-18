"""Bearer tokens for agents: Microsoft Entra ID (scheme `entra`) and OAuth 2.0 client credentials (`oauth2`)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

REFRESH_MARGIN_S = 300
OAUTH_REFRESH_MARGIN_S = 60


class TokenError(RuntimeError):
    def __init__(self, message: str, code: str = "not_allowed") -> None:
        super().__init__(message)
        self.code = code  # InvocationError code the gateway reports


class TokenProvider(Protocol):
    async def token(self, scope: str) -> str: ...


class StaticTokenProvider:
    """Fixed token for tests and local mock agents. Never use against real services."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def token(self, scope: str) -> str:
        return self._token


class EntraTokenProvider:
    """DefaultAzureCredential (selected by AZURE_TOKEN_CREDENTIALS); tokens cached until 5 minutes before expiry.

    azure-identity is imported on first use, so the platform runs without it until an agent needs Entra auth.
    """

    def __init__(self, credential: Any | None = None) -> None:
        self._credential = credential
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def token(self, scope: str) -> str:
        if cached := self._fresh(scope):
            return cached
        async with self._lock:
            if cached := self._fresh(scope):
                return cached
            credential = self._get_credential()
            try:
                access = await credential.get_token(scope)
            except Exception as exc:  # azure-identity raises several unrelated types; all mean "no token"
                raise TokenError(f"could not acquire an Entra token for '{scope}': {exc.__class__.__name__}") from exc
            self._cache[scope] = (access.token, float(access.expires_on))
            return access.token  # type: ignore[no-any-return]

    async def close(self) -> None:
        if self._credential is not None and hasattr(self._credential, "close"):
            await self._credential.close()

    def _fresh(self, scope: str) -> str | None:
        cached = self._cache.get(scope)
        if cached and cached[1] - REFRESH_MARGIN_S > time.time():
            return cached[0]
        return None

    def _get_credential(self) -> Any:
        if self._credential is None:
            try:
                from azure.identity.aio import DefaultAzureCredential  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise TokenError("Entra auth needs azure-identity: pip install -e .[azure]") from exc
            try:
                self._credential = DefaultAzureCredential(require_envvar=True)  # pyright: ignore[reportUnknownVariableType]
            except ValueError as exc:  # require_envvar: AZURE_TOKEN_CREDENTIALS is not set
                raise TokenError(f"Entra credential is not configured: {exc}") from exc
        return self._credential


class OAuthClientCredentialsProvider:
    """OAuth 2.0 client-credentials tokens (client_secret_post), cached per (token_url, client_id, scope, audience)
    until 60 seconds before `expires_in`. Tokens and secrets never appear in errors."""

    def __init__(self, http: httpx.AsyncClient, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._http = http
        self._clock = clock
        self._cache: dict[tuple[str, str, str | None, str | None], tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def token(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        *,
        scope: str | None = None,
        audience: str | None = None,
    ) -> str:
        key = (token_url, client_id, scope, audience)
        async with self._lock:
            cached = self._cache.get(key)
            if cached and cached[1] > self._clock():
                return cached[0]
            form = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
            form |= {name: value for name, value in (("scope", scope), ("audience", audience)) if value}
            try:
                response = await self._http.post(token_url, data=form, headers={"Accept": "application/json"})
            except httpx.TimeoutException as exc:
                raise TokenError("OAuth token endpoint timed out", "timeout") from exc
            except httpx.TransportError as exc:
                raise TokenError(f"OAuth token endpoint unreachable: {exc.__class__.__name__}", "unavailable") from exc
            status = response.status_code
            if status == 429 or status >= 500:
                raise TokenError(f"OAuth token endpoint returned HTTP {status}", "unavailable")
            if status != 200:
                raise TokenError(f"OAuth token endpoint returned HTTP {status}")
            try:
                body = response.json()
            except ValueError as exc:
                raise TokenError("OAuth token endpoint returned non-JSON") from exc
            access = body.get("access_token") if isinstance(body, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if not isinstance(access, str) or not access:
                raise TokenError("OAuth token endpoint returned no access_token")
            try:
                expires_in = float(body.get("expires_in") or 3600)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportOptionalMemberAccess]
            except (TypeError, ValueError):
                expires_in = 3600.0
            self._cache[key] = (access, self._clock() + expires_in - OAUTH_REFRESH_MARGIN_S)
            return access
