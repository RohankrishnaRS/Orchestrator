# pyright: basic
"""Microsoft 365 Agents SDK application: turns a Teams / Copilot message into a UEOA goal and replies with the result.

Thin by design: no planning or agent selection here - UEOA is the orchestrator. This module only
(1) extracts the user text, (2) calls POST /api/v1/goals through UeoaClient, (3) renders the reply.
"""

from __future__ import annotations

import logging
import re
import uuid
from os import environ

from microsoft_agents.hosting.aiohttp import CloudAdapter
from microsoft_agents.hosting.core import AgentApplication, MemoryStorage, TurnContext, TurnState

from ueoa_copilot_adapter.ueoa_client import UeoaClient, UeoaClientError

logger = logging.getLogger("ueoa_copilot_adapter")

WELCOME = (
    "Hi, I'm the **UEOA Orchestrator**. Describe a goal and I'll plan it, run it through the governed agent "
    "registry and return the result. Try: *Write a Python function that checks whether a number is prime.*\n\n"
    "Commands: `/help`, `/status`"
)


def build_adapter() -> tuple[CloudAdapter, object | None]:
    """Authenticated adapter when the Entra app settings exist (Azure Bot), anonymous adapter for local Playground."""
    if environ.get("CONNECTIONS__SERVICE_CONNECTION__SETTINGS__CLIENTID"):
        from microsoft_agents.activity import load_configuration_from_env
        from microsoft_agents.authentication.msal import MsalConnectionManager

        config = load_configuration_from_env(environ)
        connections = MsalConnectionManager(**config)
        logger.info("adapter: authenticated mode (Entra app configured)")
        return CloudAdapter(connection_manager=connections), connections.get_default_connection_configuration()
    logger.warning("adapter: ANONYMOUS mode - local development with Agents Playground only")
    return CloudAdapter(), None


ADAPTER, AUTH_CONFIGURATION = build_adapter()
AGENT_APP = AgentApplication[TurnState](storage=MemoryStorage(), adapter=ADAPTER)
UEOA = UeoaClient()


def correlation_id_for(context: TurnContext) -> str:
    conversation = getattr(getattr(context.activity, "conversation", None), "id", "") or "local"
    return f"teams-{conversation[-8:]}-{uuid.uuid4().hex[:8]}"


async def process_user_message(text: str, correlation_id: str) -> str:
    """Goal in, Markdown reply out. Errors are rendered, never raised to the channel."""
    try:
        outcome = await UEOA.run_goal(goal=text, correlation_id=correlation_id)
    except UeoaClientError as exc:
        logger.warning("goal failed [%s]: %s %s", exc.correlation_id, exc.code, exc.message)
        return (
            f"UEOA could not complete the request (**{exc.code}**): {exc.message}\n\n"
            f"_Correlation ID: `{exc.correlation_id}`_"
        )
    if not outcome.finished:
        return (
            f"Still working on it - run `{outcome.run_id}` is **{outcome.status}**. Ask me again in a moment.\n\n"
            f"_Correlation ID: `{outcome.correlation_id}`_"
        )
    body = outcome.result_markdown.strip() or "The run finished without a text result."
    return f"{body}\n\n_Run `{outcome.run_id}` - Correlation `{outcome.correlation_id}`_"


@AGENT_APP.conversation_update("membersAdded")
async def on_members_added(context: TurnContext, _state: TurnState) -> None:
    await context.send_activity(WELCOME)


@AGENT_APP.message(re.compile(r"^/help$", re.IGNORECASE))
async def on_help(context: TurnContext, _state: TurnState) -> None:
    await context.send_activity(WELCOME)


@AGENT_APP.message(re.compile(r"^/status$", re.IGNORECASE))
async def on_status(context: TurnContext, _state: TurnState) -> None:
    cid = correlation_id_for(context)
    try:
        health = await UEOA.health()
        store = health.get("store") or {}
        await context.send_activity(
            f"UEOA at `{UEOA.settings.base_url}` is **{health.get('status', '?')}** "
            f"(planner: {health.get('planner')}, store: {store.get('backend')})."
        )
    except UeoaClientError as exc:
        await context.send_activity(f"UEOA unreachable: {exc.message} _(correlation `{cid}`)_")


@AGENT_APP.activity("message")
async def on_message(context: TurnContext, _state: TurnState) -> None:
    text = (context.activity.text or "").strip()
    if not text:
        await context.send_activity("Please send a text goal.")
        return
    cid = correlation_id_for(context)
    logger.info("goal received [%s]: %.80s", cid, text)
    await context.send_activity(f"Working on it... _(correlation `{cid}`)_")
    await context.send_activity(await process_user_message(text, cid))


@AGENT_APP.error
async def on_error(context: TurnContext, error: Exception) -> None:
    logger.exception("unhandled error in turn: %s", error)
    await context.send_activity("The agent hit an unexpected error. Please try again.")