from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app
from ueoa.persistence.settings import StoreSettings
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa.registry.seed import load_capabilities
from ueoa_sdk.host import build_card, create_agent_app

pytestmark = pytest.mark.live
FIXTURE_CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "catalog"


def deterministic(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("item", {})
    return {"id": str(item.get("id", "deterministic")), "reference": "applied"}


async def remote_deterministic(payload: dict[str, Any], _context: Any) -> dict[str, Any]:
    return deterministic(payload)


@pytest.fixture(scope="module", autouse=True)
def postgres_available() -> Iterator[None]:
    url = os.environ.get("UEOA_STORE_URL", "").strip()
    if not url:
        pytest.skip("UEOA_STORE_URL is not set")

    async def check() -> bool:
        from ueoa.persistence.postgres.engine import Database

        database = Database(StoreSettings.from_env())
        try:
            return await database.ping()
        finally:
            await database.dispose()

    if not asyncio.run(check()):
        pytest.skip("configured PostgreSQL database is unreachable")
    yield


def _card(agent_id: str, capability: str, effects: str, *, local: bool) -> dict[str, Any]:
    params: dict[str, Any] = {
        "id": agent_id,
        "owner": "persistence-test",
        "kind": "tool",
        "implementation": "deterministic",
        "binding": "local" if local else "a2a",
        "implements": [f"{capability}@1"],
        "effects": effects,
        "timeout_s": 5,
    }
    if local:
        params["entrypoint"] = f"{__name__}:deterministic"
    return {
        "name": agent_id,
        "description": "Deterministic persistence integration test agent.",
        "version": "1.0.0",
        "capabilities": {"extensions": [{"uri": UEOA_EXTENSION_URI, "params": params}]},
        "skills": [{"id": "test", "name": "Test", "description": "Test persistence."}],
    }


def _write_catalog(catalog: Path, suffix: str) -> tuple[str, str]:
    capabilities_dir = catalog / "capabilities"
    implementations_dir = catalog / "implementations"
    capabilities_dir.mkdir(parents=True)
    implementations_dir.mkdir()
    source = load_capabilities(FIXTURE_CATALOG / "capabilities")
    read_contract = source["item.ingest"].model_copy(update={"id": f"persist.{suffix}.ingest"})
    write_contract = source["action.apply"].model_copy(update={"id": f"persist.{suffix}.apply"})
    contracts = [item.model_dump(mode="json") for item in (read_contract, write_contract)]
    (capabilities_dir / "test.yaml").write_text(yaml.safe_dump(contracts), encoding="utf-8")
    seed_cards = [
        _card(f"persist-{suffix}-local", read_contract.id, "read", local=True),
        _card(f"persist-{suffix}-writer", write_contract.id, "write", local=True),
    ]
    (implementations_dir / "test.yaml").write_text(yaml.safe_dump(seed_cards), encoding="utf-8")
    return read_contract.id, write_contract.id


def _db_cleanup(suffix: str, capability_ids: list[str], run_ids: list[str]) -> None:
    from sqlalchemy import delete

    from ueoa.persistence.postgres.engine import Database
    from ueoa.persistence.postgres.models import (
        AuditRow,
        CapabilityRow,
        ImplementationRow,
        RunRow,
        WorkflowRow,
    )

    async def clean() -> None:
        database = Database(StoreSettings.from_env())
        try:
            async with database.session() as session:
                await session.execute(delete(AuditRow).where(AuditRow.capability.in_(capability_ids)))
                if run_ids:
                    await session.execute(
                        delete(RunRow).where(RunRow.id.in_([uuid.UUID(run_id) for run_id in run_ids]))
                    )
                await session.execute(delete(RunRow).where(RunRow.goal.like(f"persist-{suffix}-%")))
                await session.execute(delete(WorkflowRow).where(WorkflowRow.id.like(f"persist-{suffix}-%")))
                await session.execute(
                    delete(ImplementationRow).where(ImplementationRow.agent_id.like(f"persist-{suffix}-%"))
                )
                await session.execute(
                    delete(CapabilityRow).where(CapabilityRow.id.in_(capability_ids))
                )
        finally:
            await database.dispose()

    asyncio.run(clean())


def test_postgres_persists_registry_workflows_runs_audit_and_pending_approvals(tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:12]
    read_capability, write_capability = _write_catalog(tmp_path, suffix)
    workflow_id = f"persist-{suffix}-flow"
    approval_workflow_id = f"persist-{suffix}-approval"
    run_ids: list[str] = []
    base_card = _card(f"persist-{suffix}-remote", read_capability, "read", local=False)
    agent = create_agent_app(
        base_card,
        {read_capability: remote_deterministic},
        "http://agent.test",
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=agent))
    settings = Settings(
        catalog_dir=tmp_path,
        allowed_hosts=("agent.test",),
        local_tool_prefixes=(__name__,),
        planner_enabled=False,
    )
    try:
        with TestClient(create_app(settings, http)) as client:
            registered = client.post(
                "/api/v1/implementations/cards",
                json=build_card("http://agent.test", base_card),
                headers={"X-Demo-Role": "admin"},
            )
            assert registered.status_code == 201, registered.text
            key = registered.json()["key"]
            approved = client.post(
                f"/api/v1/implementations/{key}/approve", json={"priority": 20}, headers={"X-Demo-Role": "admin"}
            )
            assert approved.status_code == 200, approved.text
            assert client.get(f"/api/v1/implementations/{key}").json()["status"] == "approved"

            workflow = {
                "id": workflow_id,
                "version": "1.0.0",
                "goal": f"persist-{suffix}-deterministic run",
                "inputs": {"item": {"type": "object"}},
                "nodes": [{"id": "ingest", "capability": read_capability, "inputs": {"item": "$inputs.item"}}],
                "outputs": {"id": "$nodes.ingest.output.id"},
            }
            saved = client.put(
                f"/api/v1/workflows/{workflow_id}",
                json={"name": "Persistence test", "workflow": workflow},
            )
            assert saved.status_code == 200, saved.text

            response = client.post(
                "/api/v1/runs",
                json={"workflow_id": workflow_id, "inputs": {"item": {"id": suffix}}, "wait_s": 10},
            )
            assert response.status_code == 201, response.text
            completed = response.json()
            assert completed["status"] == "completed"
            run_ids.append(completed["id"])

            approval_workflow = {
                "id": approval_workflow_id,
                "version": "1.0.0",
                "goal": f"persist-{suffix}-pending approval",
                "inputs": {"payload": {"type": "object"}},
                "nodes": [
                    {
                        "id": "apply",
                        "capability": write_capability,
                        "effect": "write",
                        "approval": {"required": True, "binds_to": "payload_hash"},
                        "inputs": {"payload": "$inputs.payload"},
                    }
                ],
            }
            saved_approval = client.put(
                f"/api/v1/workflows/{approval_workflow_id}",
                json={"workflow": approval_workflow},
            )
            assert saved_approval.status_code == 200, saved_approval.text
            waiting_response = client.post(
                "/api/v1/runs",
                json={
                    "workflow_id": approval_workflow_id,
                    "inputs": {"payload": {"id": suffix}},
                    "wait_s": 10,
                },
            )
            assert waiting_response.status_code == 201, waiting_response.text
            waiting = waiting_response.json()
            assert waiting["status"] == "waiting"
            run_ids.append(waiting["id"])

        with TestClient(create_app(settings, http)) as restarted:
            assert restarted.get(f"/api/v1/implementations/{key}").json()["status"] == "approved"
            assert restarted.get(f"/api/v1/workflows/{workflow_id}").status_code == 200
            assert restarted.get(f"/api/v1/runs/{completed['id']}").json()["status"] == "completed"
            pending = restarted.get("/api/v1/approvals", headers={"X-Demo-Role": "admin"}).json()
            assert any(approval["run_id"] == waiting["id"] for approval in pending)
            audit = restarted.get("/api/v1/audit", headers={"X-Demo-Role": "admin"}).json()
            assert sum(entry["capability"] == read_capability for entry in audit) == 1

        from sqlalchemy import select

        from ueoa.persistence.postgres.engine import Database
        from ueoa.persistence.postgres.models import RunNodeRow

        async def run_node_count() -> int:
            database = Database(StoreSettings.from_env())
            try:
                async with database.session() as session:
                    rows = await session.execute(
                        select(RunNodeRow).where(RunNodeRow.run_id == uuid.UUID(completed["id"]))
                    )
                    return len(list(rows.scalars()))
            finally:
                await database.dispose()

        assert asyncio.run(run_node_count()) == 1
    finally:
        asyncio.run(http.aclose())
        _db_cleanup(suffix, [read_capability, write_capability], run_ids)
