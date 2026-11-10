"""Foundation-gateway LLM client: token flow, secure route, 401 refresh, JSON retry - against a mock transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ueoa.jsontext import extract_json_object
from ueoa.llm import LLMRateLimitedError, LLMUnavailableError
from ueoa_sdk.llm import FoundationLLM, FoundationSettings, LLMError

pytestmark = pytest.mark.anyio
URL = "https://gateway.test/foundation-api/api/agentic_openai/generate_response"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _settings(key: str = "foundation-key") -> FoundationSettings:
    return FoundationSettings(URL, "model-x", "agent-x", "proj", "proj", "DEV", key)


def _chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_secure_route_with_token_refresh_and_json_retry() -> None:
    seen: list[tuple[str, str | None]] = []
    replies = iter([httpx.Response(401), _chat("not json"), _chat('```json\n{"ok": true}\n```')])
    tokens = iter(["t1", "t2"])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        if request.url.path.endswith("/get_token"):
            assert json.loads(request.content) == {"password": "foundation-key"}
            return httpx.Response(200, json={"access_token": next(tokens), "token_type": "bearer"})
        body = json.loads(request.content)
        assert body["metadata"] == {"user_id": "ueoa", "chain_id": "inv-1", "parent_id": "agent-a"}
        return next(replies)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = FoundationLLM(_settings(), agent_name="agent-a", http=http)
        assert await llm.complete_json("sys", "user", chain_id="inv-1") == {"ok": True}
    assert seen == [
        ("/foundation-api/get_token", None),
        ("/foundation-api/api/agentic_openai/generate_secure_response", "Bearer t1"),
        ("/foundation-api/get_token", None),
        ("/foundation-api/api/agentic_openai/generate_secure_response", "Bearer t2"),
        ("/foundation-api/api/agentic_openai/generate_secure_response", "Bearer t2"),
    ]


async def test_gateway_errors_never_leak_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal secret detail")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = FoundationLLM(_settings(key=""), agent_name="a", http=http)
        with pytest.raises(LLMError) as error:
            await llm.complete("s", "u")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(("status", "error"), [(429, LLMRateLimitedError), (503, LLMUnavailableError), (400, LLMError)])
async def test_status_codes_map_to_retryable_errors(status: int, error: type[LLMError]) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(status))) as http:
        llm = FoundationLLM(_settings(key=""), agent_name="a", http=http)
        with pytest.raises(LLMError) as raised:
            await llm.complete_json("s", "u")
    assert type(raised.value) is error


async def test_token_rate_limit_and_transport_errors_are_retryable() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(429))) as http:
        with pytest.raises(LLMRateLimitedError, match="get_token"):
            await FoundationLLM(_settings(), agent_name="a", http=http).complete("s", "u")

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as http:
        with pytest.raises(LLMUnavailableError, match="unreachable"):
            await FoundationLLM(_settings(key=""), agent_name="a", http=http).complete_json("s", "u")


async def test_one_deadline_covers_the_json_retry() -> None:
    calls = 0

    async def slow(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.15)
        return _chat("not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as http:
        llm = FoundationLLM(_settings(key=""), agent_name="a", http=http, deadline_s=0.2)
        with pytest.raises(LLMUnavailableError, match="deadline"):
            await llm.complete_json("s", "u")
    assert calls == 2


def test_settings_report_missing_names(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("UEOA_LLM_URL", "UEOA_LLM_MODEL", "UEOA_LLM_AGENT_ID", "UEOA_LLM_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match="UEOA_LLM_URL"):
        FoundationSettings.from_env()


@pytest.mark.parametrize(
    ("text", "expected"),
    [('{"a": 1}', {"a": 1}), ('Sure:\n```json\n{"a": 2}\n```', {"a": 2}), ('x {"a": 3} y', {"a": 3}), ("[1]", None)],
)
def test_extract_json_object(text: str, expected: dict[str, int] | None) -> None:
    assert extract_json_object(text) == expected
