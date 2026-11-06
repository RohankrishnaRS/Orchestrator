from typing import Any

from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app

USER = {"X-Demo-Role": "user"}
WORKFLOW: dict[str, Any] = {
    "id": "studio-flow",
    "version": "0.1.0",
    "goal": "Collect a review input",
    "allowed_tasks": ["review"],
    "inputs": {"input": {"type": "object"}},
    "nodes": [{"id": "review", "type": "human_input", "input_schema": {"type": "object"}}],
    "outputs": {},
}


class PromptLLM:
    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        assert "professional" in system.lower()
        assert "Order each step" in user
        return {"prompt": "A professional, implementation-ready workflow brief."}


def test_studio_route_and_publish_attestation() -> None:
    with TestClient(create_app(Settings(catalog_dir=None, planner_enabled=False))) as client:
        assert client.get("/create/workflow").status_code == 200
        assert client.get("/create/saved-workflows").status_code == 200
        assert client.get("/create/prompt").status_code == 200
        saved = client.put(
            "/api/v1/workflows/studio-flow",
            json={"name": "Studio flow", "description": WORKFLOW["goal"], "workflow": WORKFLOW},
            headers=USER,
        )
        assert saved.status_code == 200, saved.text

        published = client.post(
            "/api/v1/workflows/studio-flow/publish",
            json={"attest": True},
            headers=USER,
        )
        assert published.status_code == 200, published.text
        assert published.json()["published"] is True
        assert published.json()["attestation"]["status"] == "attested"
        version_id = published.json()["id"]
        assert published.json()["version_ref"] == "studio-flow@0.1.0"
        assert any(
            item["id"] == version_id
            for item in client.get("/api/v1/workflows", headers=USER).json()
        )
        assert client.post(
            "/api/v1/workflows/studio-flow/publish",
            json={"attest": True},
            headers=USER,
        ).status_code == 409
        assert client.get(f"/api/v1/workflows/{version_id}", headers=USER).status_code == 200

        run = client.post(
            "/api/v1/runs",
            json={"workflow_id": version_id, "inputs": {"input": {}}, "wait_s": 5},
            headers=USER,
        )
        assert run.status_code == 201, run.text
        assert run.json()["status"] == "waiting"

        update = client.put(
            f"/api/v1/workflows/{version_id}",
            json={"workflow": WORKFLOW},
            headers=USER,
        )
        assert update.status_code == 409
        delete = client.delete(f"/api/v1/workflows/{version_id}", headers=USER)
        assert delete.status_code == 409

        updated_workflow = {**WORKFLOW, "version": "0.1.1"}
        saved_new_version = client.put(
            "/api/v1/workflows/studio-flow",
            json={"name": "Studio flow", "workflow": updated_workflow},
            headers=USER,
        )
        assert saved_new_version.status_code == 200, saved_new_version.text
        next_version = client.post(
            "/api/v1/workflows/studio-flow/publish",
            json={"attest": False},
            headers=USER,
        )
        assert next_version.status_code == 200, next_version.text
        assert next_version.json()["version_ref"] == "studio-flow@0.1.1"


def test_allowed_tasks_required_for_studio_workflows() -> None:
    workflow = {**WORKFLOW, "id": "missing-tasks", "allowed_tasks": []}
    with TestClient(create_app(Settings(catalog_dir=None, planner_enabled=False))) as client:
        response = client.post("/api/v1/workflows/validate", json={"workflow": workflow}, headers=USER)
    assert response.status_code == 200
    assert response.json()["ok"] is False


def test_invalid_workflow_can_be_saved_and_resumed_as_a_draft() -> None:
    workflow = {**WORKFLOW, "id": "unfinished-draft", "allowed_tasks": []}
    with TestClient(create_app(Settings(catalog_dir=None, planner_enabled=False))) as client:
        saved = client.put(
            "/api/v1/workflows/unfinished-draft",
            json={"name": "Unfinished draft", "workflow": workflow},
            headers=USER,
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["valid"] is False
        resumed = client.get("/api/v1/workflows/unfinished-draft", headers=USER)
        assert resumed.status_code == 200
        assert resumed.json()["workflow"]["id"] == "unfinished-draft"


def test_prompt_builder_refines_brief_with_configured_llm() -> None:
    with TestClient(
        create_app(Settings(catalog_dir=None, planner_enabled=False), agent_llm=PromptLLM())
    ) as client:
        response = client.post(
            "/api/v1/workflows/prompt",
            json={
                "description": "Review incoming support requests and summarize them.",
                "audience": "Support team",
                "inputs": "Request text",
                "steps": "Order each step",
                "constraints": "Do not make changes",
                "output": "Short summary",
            },
            headers=USER,
        )
    assert response.status_code == 200, response.text
    assert response.json() == {"prompt": "A professional, implementation-ready workflow brief."}


def test_workflow_draft_reports_missing_approved_capabilities() -> None:
    with TestClient(
        create_app(Settings(catalog_dir=None, planner_enabled=False), planner_llm=PromptLLM())
    ) as client:
        response = client.post(
            "/api/v1/workflows/draft",
            json={"goal": "Review incoming support requests"},
            headers=USER,
        )
    assert response.status_code == 200, response.text
    draft = response.json()
    assert draft["ok"] is False
    assert draft["workflow"] is None
    assert draft["issues"] == ["no read capability with an approved implementation"]
