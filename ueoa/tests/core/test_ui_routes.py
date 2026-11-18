"""Console page URLs should serve the same SPA and preserve API/static routes."""

import re
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from ueoa.api.app import CONSOLE_PAGE_PATHS, Settings, create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    app = create_app(Settings(catalog_dir=None, planner_enabled=False), http, planner_llm=None)
    with TestClient(app) as test_client:
        yield test_client


def test_console_page_paths_serve_the_same_html(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200

    for path in CONSOLE_PAGE_PATHS:
        response = client.get(path)
        assert response.status_code == 200
        assert response.content == home.content


def test_api_assets_and_unknown_paths_keep_their_behavior(client: TestClient) -> None:
    assert client.get("/api/v1/health").status_code == 200

    home = client.get("/")
    scripts = re.findall(r'<script[^>]*\bsrc=["\']([^"\']+)', home.text)
    assert scripts
    for path in scripts:
        response = client.get(path)
        assert response.status_code == 200
        assert response.content

    assert client.get("/nope").status_code == 404
