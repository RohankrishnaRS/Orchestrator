# pyright: basic
"""aiohttp host for the adapter: POST /api/messages (Agents SDK) and GET /api/health."""

from __future__ import annotations

import json
import logging
from os import environ

from aiohttp.web import Application, Request, Response, run_app
from microsoft_agents.hosting.aiohttp import CloudAdapter, jwt_authorization_middleware, start_agent_process
from microsoft_agents.hosting.core import AgentApplication

from ueoa_copilot_adapter.agent import ADAPTER, AGENT_APP, AUTH_CONFIGURATION, UEOA

logger = logging.getLogger("ueoa_copilot_adapter.server")


async def messages(request: Request) -> Response:
    agent: AgentApplication = request.app["agent_app"]
    adapter: CloudAdapter = request.app["adapter"]
    return await start_agent_process(request, agent, adapter)


async def health(_: Request) -> Response:
    payload = {
        "status": "ok",
        "mode": "authenticated" if environ.get("CONNECTIONS__SERVICE_CONNECTION__SETTINGS__CLIENTID") else "anonymous-dev",
        "ueoa_base_url": UEOA.settings.base_url,
    }
    return Response(text=json.dumps(payload), content_type="application/json")


def create_web_app() -> Application:
    app = Application(middlewares=[jwt_authorization_middleware])
    app.router.add_post("/api/messages", messages)
    app.router.add_get("/api/messages", lambda _: Response(status=200))
    app.router.add_get("/api/health", health)
    app["agent_configuration"] = AUTH_CONFIGURATION
    app["agent_app"] = AGENT_APP
    app["adapter"] = ADAPTER
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("microsoft_agents").setLevel(logging.INFO)
    port = int(environ.get("PORT", "3978"))
    host = "0.0.0.0" if environ.get("WEBSITE_SITE_NAME") else "localhost"  # Azure App Service sets WEBSITE_SITE_NAME
    logger.info("UEOA Copilot adapter listening on %s:%d (UEOA at %s)", host, port, UEOA.settings.base_url)
    run_app(create_web_app(), host=host, port=port)