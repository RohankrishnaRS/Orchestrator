# pyright: basic
"""Goals API contract: correlation id, error model, planner-unavailable and not-found paths."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app

ADMIN = {"X-Demo-Role": "admin"}


def client() -> TestClient:
    return TestClient(create_app(Settings(planner_enabled=False)))


def test_planner_unavailable_uses_error_model_and_echoes_correlation_id() -> None:
    with client() as c:
        r = c.post("/api/v1/goals", json={"goal": "do something"}, headers={**ADMIN, "X-Correlation-Id": "abc-123"})
    assert r.status_code == 503
    assert r.headers["X-Correlation-Id"] == "abc-123"
    body = r.json()
    assert body["code"] == "planner_unavailable"
    assert body["correlation_id"] == "abc-123"
    assert set(body) >= {"code", "message", "correlation_id"}


def test_correlation_id_is_generated_when_absent() -> None:
    with client() as c:
        r = c.post("/api/v1/goals", json={"goal": "do something"}, headers=ADMIN)
    assert r.status_code == 503
    cid = r.headers["X-Correlation-Id"]
    assert len(cid) == 36 and r.json()["correlation_id"] == cid


def test_bad_correlation_id_rejected() -> None:
    with client() as c:
        r = c.post("/api/v1/goals", json={"goal": "x"}, headers={**ADMIN, "X-Correlation-Id": "has spaces"})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_unknown_run_is_not_found_with_error_model() -> None:
    with client() as c:
        r = c.get("/api/v1/goals/does-not-exist", headers={**ADMIN, "X-Correlation-Id": "cid-9"})
    assert r.status_code == 404
    assert r.json() == {"code": "not_found", "message": "unknown run 'does-not-exist'", "correlation_id": "cid-9"}
    assert r.headers["X-Correlation-Id"] == "cid-9"


def test_request_validation() -> None:
    with client() as c:
        assert c.post("/api/v1/goals", json={"goal": ""}, headers=ADMIN).status_code == 422
        assert c.post("/api/v1/goals", json={"goal": "x", "wait_seconds": 999}, headers=ADMIN).status_code == 422


def test_openapi_documents_the_goals_tag() -> None:
    with client() as c:
        spec = c.get("/openapi.json").json()
    assert "/api/v1/goals" in spec["paths"]
    assert "/api/v1/goals/{run_id}" in spec["paths"]
    assert "Goals" in spec["paths"]["/api/v1/goals"]["post"]["tags"]