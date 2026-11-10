# pyright: basic
"""UeoaClient: success, 202 -> poll -> 200, retry on 503, error model mapping."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from ueoa_copilot_adapter.ueoa_client import UeoaClient, UeoaClientError, UeoaClientSettings

BASE = "http" + "://ueoa.test"


def client_with(handler: Any) -> UeoaClient:
    settings = UeoaClientSettings(base_url=BASE, poll_seconds=0.01, max_total_seconds=2, retries=1)
    return UeoaClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE))


def goal_payload(status: str, finished: bool, cid: str = "c-1") -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "status": status,
        "finished": finished,
        "correlation_id": cid,
        "result_markdown": "### Result\n\n- **code_snippet**: print(1)" if finished else None,
        "outputs": {},
    }


def test_success_uses_demo_header_and_correlation_id() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=goal_payload("completed", True, "abc"))

    outcome = asyncio.run(client_with(handler).run_goal("do x", correlation_id="abc"))
    assert outcome.finished and outcome.run_id == "run-1" and outcome.correlation_id == "abc"
    assert seen["headers"]["x-correlation-id"] == "abc"
    assert seen["headers"]["x-demo-role"] == "reviewer"
    assert seen["body"]["goal"] == "do x" and "wait_seconds" in seen["body"]


def test_running_then_polls_until_finished() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            return httpx.Response(202, json=goal_payload("running", False))
        return httpx.Response(200, json=goal_payload("completed", len(calls) >= 3))

    outcome = asyncio.run(client_with(handler).run_goal("long task"))
    assert outcome.finished
    assert calls[0] == "POST /api/v1/goals" and calls[1].startswith("GET /api/v1/goals/run-1")


def test_retries_on_503_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={"code": "planner_unavailable", "message": "down", "correlation_id": "c"})
        return httpx.Response(200, json=goal_payload("completed", True))

    outcome = asyncio.run(client_with(handler).run_goal("x"))
    assert outcome.finished and attempts["n"] == 2


def test_error_model_is_mapped() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"code": "forbidden", "message": "nope", "correlation_id": "c-9"})

    with pytest.raises(UeoaClientError) as info:
        asyncio.run(client_with(handler).run_goal("x", correlation_id="c-9"))
    assert info.value.code == "forbidden" and info.value.correlation_id == "c-9" and info.value.status == 403


def test_bearer_token_header_when_configured() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json=goal_payload("completed", True))

    settings = UeoaClientSettings(base_url=BASE, bearer_token="tok", retries=0)
    client = UeoaClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(client.run_goal("x"))
    assert seen["authorization"] == "Bearer tok" and "x-demo-role" not in seen