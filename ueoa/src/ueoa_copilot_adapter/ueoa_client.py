# pyright: basic
"""Client for the real UEOA Goals API (POST/GET /api/v1/goals). Replaces the demo execute_goal()."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUS = {502, 503, 504}
DEFAULT_BASE_URL = "http" + "://127.0.0.1:8000"


@dataclass(frozen=True)
class UeoaClientSettings:
    base_url: str = field(default_factory=lambda: os.environ.get("UEOA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"))
    bearer_token: str = field(default_factory=lambda: os.environ.get("UEOA_BEARER_TOKEN", ""))
    demo_role: str = field(default_factory=lambda: os.environ.get("UEOA_DEMO_ROLE", "reviewer"))
    wait_seconds: int = field(default_factory=lambda: int(os.environ.get("UEOA_WAIT_SECONDS", "90")))
    poll_seconds: float = 3.0
    max_total_seconds: float = 240.0
    retries: int = 2


@dataclass
class GoalOutcome:
    run_id: str
    status: str
    finished: bool
    correlation_id: str
    result_markdown: str
    outputs: Any = None


class UeoaClientError(Exception):
    def __init__(self, code: str, message: str, correlation_id: str, status: int | None = None):
        super().__init__(message)
        self.code, self.message, self.correlation_id, self.status = code, message, correlation_id, status


class UeoaClient:
    def __init__(self, settings: UeoaClientSettings | None = None, http: httpx.AsyncClient | None = None):
        self.settings = settings or UeoaClientSettings()
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(self.settings.wait_seconds + 30, connect=10))

    def _headers(self, correlation_id: str) -> dict[str, str]:
        headers = {"X-Correlation-Id": correlation_id, "Accept": "application/json"}
        if self.settings.bearer_token:
            headers["Authorization"] = f"Bearer {self.settings.bearer_token}"
        else:
            headers["X-Demo-Role"] = self.settings.demo_role  # local development only
        return headers

    async def run_goal(
        self, goal: str, inputs: dict[str, Any] | None = None, correlation_id: str | None = None
    ) -> GoalOutcome:
        """POST the goal, then poll until finished or max_total_seconds elapses."""
        cid = correlation_id or str(uuid.uuid4())
        body = {"goal": goal, "inputs": inputs or {}, "wait_seconds": self.settings.wait_seconds}
        response = await self._request("POST", "/api/v1/goals", cid, json=body)
        outcome = self._outcome(response.json())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.max_total_seconds
        while not outcome.finished and loop.time() < deadline:
            await asyncio.sleep(self.settings.poll_seconds)
            outcome = await self.get_goal(outcome.run_id, cid)
        return outcome

    async def get_goal(self, run_id: str, correlation_id: str) -> GoalOutcome:
        response = await self._request("GET", f"/api/v1/goals/{run_id}", correlation_id)
        return self._outcome(response.json())

    async def health(self) -> dict[str, Any]:
        response = await self._request("GET", "/api/v1/health", str(uuid.uuid4()))
        return response.json()

    async def _request(self, method: str, path: str, cid: str, **kwargs: Any) -> httpx.Response:
        url = self.settings.base_url + path
        last: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                response = await self._http.request(method, url, headers=self._headers(cid), **kwargs)
            except httpx.TransportError as exc:
                last = exc
                logger.warning("UEOA unreachable (attempt %d) [%s]: %s", attempt + 1, cid, exc.__class__.__name__)
            else:
                if response.status_code in RETRY_STATUS and attempt < self.settings.retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    raise self._error(response, cid)
                return response
            await asyncio.sleep(1.5 * (attempt + 1))
        raise UeoaClientError("upstream_unavailable", f"UEOA not reachable: {last}", cid)

    @staticmethod
    def _error(response: httpx.Response, cid: str) -> UeoaClientError:
        try:
            body = response.json()
        except ValueError:
            body = {}
        code = str(body.get("code") or f"http_{response.status_code}")
        message = str(body.get("message") or body.get("detail") or response.text[:300])
        return UeoaClientError(code, message, str(body.get("correlation_id") or cid), response.status_code)

    @staticmethod
    def _outcome(payload: dict[str, Any]) -> GoalOutcome:
        return GoalOutcome(
            run_id=str(payload["run_id"]),
            status=str(payload.get("status", "")),
            finished=bool(payload.get("finished")),
            correlation_id=str(payload.get("correlation_id", "")),
            result_markdown=str(payload.get("result_markdown") or ""),
            outputs=payload.get("outputs"),
        )

    async def aclose(self) -> None:
        await self._http.aclose()