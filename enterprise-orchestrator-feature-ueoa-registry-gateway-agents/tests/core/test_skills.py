from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app

ADMIN = {"X-Demo-Role": "admin"}
USER = {"X-Demo-Role": "user"}
CUSTOM_SKILL: dict[str, Any] = {
    "name": "Incident Triage",
    "description": "Classify an incident report and suggest next steps.",
    "instructions": "Use only the supplied incident details and state uncertainty.",
    "tags": ["operations", "triage"],
}


def _client(data_dir: Path) -> TestClient:
    return TestClient(create_app(Settings(catalog_dir=None, data_dir=data_dir)))


def test_default_skills_are_listed_and_read_only(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        defaults = client.get("/api/v1/skills", headers=USER)
        assert defaults.status_code == 200
        assert {skill["id"] for skill in defaults.json()} == {
            "document-creation",
            "summarization",
            "research",
            "data-analysis",
            "information-extraction",
            "validation",
            "content-generation",
        }
        assert client.delete("/api/v1/skills/research", headers=ADMIN).status_code == 409
        assert client.post("/api/v1/skills", json=CUSTOM_SKILL, headers=USER).status_code == 403


def test_custom_skill_crud_persists_across_app_restart(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = client.post("/api/v1/skills", json=CUSTOM_SKILL, headers=ADMIN)
        assert created.status_code == 201
        assert created.json()["id"] == "incident-triage"
        changed = {**CUSTOM_SKILL, "description": "Updated incident classification."}
        updated = client.put("/api/v1/skills/incident-triage", json=changed, headers=ADMIN)
        assert updated.status_code == 200

    with _client(tmp_path) as client:
        stored = client.get("/api/v1/skills", headers=USER).json()
        custom = next(skill for skill in stored if skill["id"] == "incident-triage")
        assert custom["description"] == "Updated incident classification."
        assert custom["instructions"] == CUSTOM_SKILL["instructions"]
        assert client.delete("/api/v1/skills/incident-triage", headers=ADMIN).status_code == 200


def test_overview_reports_live_registry_and_skill_counts(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        result = client.get("/api/v1/overview", headers=USER)
        assert result.status_code == 200
        assert result.json() == {
            "total_agents": 0,
            "active_agents": 0,
            "total_workflows": 0,
            "total_skills": 7,
            "registered_mcp_servers": 0,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "recent_activities": [],
        }
