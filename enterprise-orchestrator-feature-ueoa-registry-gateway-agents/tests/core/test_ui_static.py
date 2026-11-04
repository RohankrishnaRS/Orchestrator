"""The console must remain domain-agnostic and serve its bundled UI without external assets."""

import re
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ueoa.api.app import Settings, create_app

DOMAIN_WORDS = re.compile(r"contract|clause|legal", re.IGNORECASE)
# Acronyms match case-sensitively so ordinary words ("through", "handle") do not trip the audit.
DOMAIN_ACRONYMS = re.compile(r"\b(NDA|HR)\b")
INNER_HTML_ASSIGN = re.compile(r"\.innerHTML\s*\+?=(?!=)")
STATIC_SOURCE = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def html() -> Iterator[str]:
    settings = Settings(catalog_dir=None, planner_enabled=False)
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with TestClient(create_app(settings, http, planner_llm=None)) as client:
        response = client.get("/")
        assert response.status_code == 200
        yield response.text


def test_page_has_no_domain_words(html: str) -> None:
    assert DOMAIN_WORDS.findall(html) == []
    assert DOMAIN_ACRONYMS.findall(html) == []


def test_page_loads_nothing_external(html: str) -> None:
    assert not re.search(r"<script[^>]*\bsrc\s*=\s*[\"']https?://", html, re.IGNORECASE)
    assert not re.search(r"<link[^>]*\bhref\s*=\s*[\"']?(https?:)?//", html, re.IGNORECASE)
    assert not re.search(r"\b(src|href)\s*=\s*[\"']?(https?:)?//", html, re.IGNORECASE)
    assert "@import" not in html


def test_orchestrate_section_exists(html: str) -> None:
    assert 'id="orchestrate"' in html
    assert re.search(r"<h2[^>]*>Orchestrate</h2>", html)
    assert 'aria-live="polite"' in html


def test_console_bundle_is_served_locally(html: str) -> None:
    scripts = re.findall(r"<script[^>]*\bsrc=[\"']([^\"']+)", html, re.IGNORECASE)
    assert scripts
    assert all(path.startswith("/assets/") for path in scripts)
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with TestClient(create_app(Settings(catalog_dir=None, planner_enabled=False), http, planner_llm=None)) as client:
        for path in scripts:
            response = client.get(path)
            assert response.status_code == 200
            assert response.content


def test_console_uses_framework_escaping() -> None:
    source = (STATIC_SOURCE / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "dangerouslySetInnerHTML" not in source
    assert not INNER_HTML_ASSIGN.search(source)
