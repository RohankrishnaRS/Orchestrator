from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app

ADMIN = {"X-Demo-Role": "admin"}
USER = {"X-Demo-Role": "user"}
MCP_SERVER: dict[str, Any] = {
    "name": "Internal Tools",
    "url": "https://mcp.example.com/mcp",
    "description": "Internal development tools.",
}


def _client(data_dir: Path) -> TestClient:
    return TestClient(create_app(Settings(catalog_dir=None, data_dir=data_dir)))


def test_register_list_delete_mcp_server_persists(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = client.post("/api/v1/mcp-servers", json=MCP_SERVER, headers=ADMIN)
        assert created.status_code == 201
        assert created.json()["id"] == "internal-tools"
        assert created.json()["transport"] == "streamable-http"
        assert client.get("/api/v1/mcp-servers", headers=USER).json() == [created.json()]
        assert client.get("/api/v1/overview", headers=USER).json()["registered_mcp_servers"] == 1
        assert client.post("/api/v1/mcp-servers", json=MCP_SERVER, headers=ADMIN).status_code == 409

    with _client(tmp_path) as client:
        assert client.get("/api/v1/mcp-servers", headers=USER).json()[0]["name"] == "Internal Tools"
        assert client.delete("/api/v1/mcp-servers/internal-tools", headers=ADMIN).status_code == 200
        assert client.get("/api/v1/mcp-servers", headers=USER).json() == []


def test_mcp_server_catalog_requires_admin_for_changes(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.post("/api/v1/mcp-servers", json=MCP_SERVER, headers=USER).status_code == 403
        assert client.delete("/api/v1/mcp-servers/anything", headers=USER).status_code == 403


def test_mcp_server_url_rejects_embedded_credentials_and_invalid_urls(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        credentials = {**MCP_SERVER, "url": "https://user:password@mcp.example.com/mcp"}
        query_credentials = {**MCP_SERVER, "url": "https://mcp.example.com/mcp?access_token=secret"}
        invalid = {**MCP_SERVER, "url": "file:///tmp/mcp"}
        assert client.post("/api/v1/mcp-servers", json=credentials, headers=ADMIN).status_code == 422
        assert client.post("/api/v1/mcp-servers", json=query_credentials, headers=ADMIN).status_code == 422
        assert client.post("/api/v1/mcp-servers", json=invalid, headers=ADMIN).status_code == 422
