import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app
from ueoa.llm import AzureOpenAIJsonLLM
from ueoa.registry.service import RegistryError

ADMIN = {"X-Demo-Role": "admin"}
USER = {"X-Demo-Role": "user"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedLLM:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.system = ""
        self.user = ""

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        self.system = system
        self.user = user
        return self.reply


def _generated_config() -> dict[str, Any]:
    return {
        "description": "Answers product questions from provided details.",
        "capability_description": "Answer product questions.",
        "use_when": "A customer asks about a product.",
        "instructions": "Answer only from supplied product information.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string", "default": "What is this product?"}},
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        },
    }


def test_agent_description_generates_reviewable_definition_with_user_name() -> None:
    llm = ScriptedLLM(_generated_config())
    with TestClient(create_app(Settings(catalog_dir=None), planner_llm=llm)) as client:
        response = client.post(
            "/api/v1/agents/generate",
            json={"name": "Product Helper", "description": "Answer product questions based on supplied details."},
            headers=ADMIN,
        )
        assert response.status_code == 200, response.text
        definition = response.json()
        assert definition["name"] == "Product Helper"
        assert definition["id"] == "agent-product-helper"
        assert definition["capability"]["id"] == "agent.product_helper"
        assert definition["instructions"] == _generated_config()["instructions"]
        assert client.post(
            "/api/v1/agents/preview",
            json={"definition": definition},
            headers=ADMIN,
        ).status_code == 200
        assert "Product Helper" in llm.user


def test_agent_generation_requires_admin_and_valid_llm_output() -> None:
    with TestClient(create_app(Settings(catalog_dir=None), planner_llm=ScriptedLLM(_generated_config()))) as client:
        payload = {"name": "A", "description": "Do something helpful."}
        assert client.post("/api/v1/agents/generate", json=payload, headers=USER).status_code == 403

    invalid_llm = ScriptedLLM({"instructions": "incomplete"})
    with TestClient(create_app(Settings(catalog_dir=None), planner_llm=invalid_llm)) as client:
        response = client.post("/api/v1/agents/generate", json=payload, headers=ADMIN)
        assert response.status_code == 502


def test_agent_generation_reports_missing_azure_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with TestClient(create_app(Settings(catalog_dir=None))) as client:
        response = client.post(
            "/api/v1/agents/generate",
            json={"name": "A", "description": "Do something helpful."},
            headers=ADMIN,
        )
    assert response.status_code == 503
    assert "Set AZURE_OPENAI_API_KEY and restart the server." in response.json()["detail"]


def test_dotenv_loading_preserves_process_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ueoa.api.app import _load_environment

    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AZURE_OPENAI_API_KEY=dotenv-test-key\nAZURE_OPENAI_ENDPOINT=https://dotenv.example.test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://process.example.test")

    _load_environment(env_file)

    assert os.environ["AZURE_OPENAI_API_KEY"] == "dotenv-test-key"
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://process.example.test"


def test_hosted_agent_registration_uses_the_running_application_url(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM(_generated_config())
    with TestClient(
        create_app(Settings(catalog_dir=None, public_base_url=""), planner_llm=llm),
        base_url="http://127.0.0.1:8766",
    ) as client:
        services = client.app.state.services  # type: ignore[attr-defined]
        registered_urls: list[str] = []

        async def fail_registration(url: str, overlay: dict[str, Any] | None = None) -> Any:
            registered_urls.append(url)
            raise RegistryError("could not fetch agent card: ConnectError")

        monkeypatch.setattr(services.registry, "register_from_url", fail_registration)
        response = client.post(
            "/api/v1/agents",
            json={
                "definition": {
                    "id": "agent-support-helper",
                    "name": "Support Helper",
                    "description": "Answers support questions from provided information.",
                    "template": "instruction",
                    "capability": {
                        "id": "agent.support_helper",
                        "description": "Answer support questions.",
                        "use_when": "A support question needs an answer.",
                    },
                    "instructions": "Answer only using the supplied information.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"question": {"type": "string", "default": "Help"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "approve": True,
                }
            },
            headers=ADMIN,
        )
        assert response.status_code == 201, response.text
        expected_root = "http://127.0.0.1:8766/hosted/agent-support-helper"
        assert registered_urls == [expected_root]
        assert response.json()["card"]["supportedInterfaces"][0]["url"] == f"{expected_root}/a2a"
        assert response.json()["registration_error"] == "could not fetch agent card: ConnectError"
        retry = client.post("/api/v1/agents/agent-support-helper/register", headers=ADMIN)
        assert retry.status_code == 502
        assert registered_urls == [expected_root, expected_root]
        assert "Agent registration failed" in retry.json()["detail"]


@pytest.mark.anyio
async def test_azure_openai_json_client_uses_server_side_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = AzureOpenAIJsonLLM(
            "https://azure.example.test",
            "test-deployment",
            "test-only-secret",
            http=client,
        )
        assert await llm.complete_json("system", "prompt") == {"ok": True}

    assert requests[0].url.path.endswith("/openai/deployments/test-deployment/chat/completions")
    assert requests[0].url.params["api-version"] == "2024-10-21"
    assert requests[0].headers["api-key"] == "test-only-secret"
    assert "test-only-secret" not in str(requests[0].url)


@pytest.mark.anyio
async def test_azure_openai_environment_uses_project_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only-secret")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    client = httpx.AsyncClient()
    try:
        llm = AzureOpenAIJsonLLM.from_env(client)
        assert llm is not None
        assert "secureapi-aoai-north-central-us.openai.azure.com" in llm._url
        assert "/deployments/gpt-4o/" in llm._url
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_application_does_not_fallback_to_legacy_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from ueoa.api.app import _build_agent_llm

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("UEOA_LLM_URL", "https://legacy.example.test/api/agentic_openai/generate_response")
    monkeypatch.setenv("UEOA_LLM_MODEL", "legacy")
    monkeypatch.setenv("UEOA_LLM_AGENT_ID", "legacy-agent")
    monkeypatch.setenv("UEOA_LLM_PROJECT_ID", "legacy-project")
    client = httpx.AsyncClient()
    try:
        assert _build_agent_llm(None, client) is None
    finally:
        await client.aclose()
