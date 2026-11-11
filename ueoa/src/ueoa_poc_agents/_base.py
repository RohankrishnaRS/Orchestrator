# pyright: basic
"""Shared plumbing for the POC agents: document loading, card building, serving."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn

from ueoa.a2a import InvocationContext
from ueoa.registry.models import UEOA_EXTENSION_URI
from ueoa_poc_agents import OWNER
from ueoa_sdk.host import create_agent_app

log = logging.getLogger("ueoa_poc_agents")
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MAX_DOCUMENT_CHARS = 60_000


def load_document_text(payload: dict[str, Any]) -> str:
    """Accept document.text, document.sample (file in samples/) or top-level document_text."""
    text = payload.get("document_text")
    if isinstance(text, str) and text.strip():
        return text[:MAX_DOCUMENT_CHARS]
    document = payload.get("document")
    sample: Any = payload.get("sample")
    if isinstance(document, dict):
        inline = document.get("text")
        if isinstance(inline, str) and inline.strip():
            return inline[:MAX_DOCUMENT_CHARS]
        sample = document.get("sample")
    if isinstance(sample, str) and sample:
        path = SAMPLES_DIR / Path(sample).name
        if not path.is_file():
            raise ValueError(f"unknown sample {sample!r}")
        return path.read_text(encoding="utf-8-sig", errors="replace")[:MAX_DOCUMENT_CHARS]
    raise ValueError("provide 'document.text', 'document.sample' or 'document_text'")


def poc_card(
    *,
    agent_id: str,
    name: str,
    description: str,
    capability: str,
    skill_description: str,
    examples: list[str],
    tags: list[str],
    implementation: str = "deterministic",
    timeout_s: int = 60,
    version: str = "1.0.0",
) -> dict[str, Any]:
    cap_id, _, major = capability.partition("@")
    return {
        "name": name,
        "description": description,
        "version": version,
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [{"id": cap_id, "name": name, "description": skill_description, "tags": tags, "examples": examples}],
        "capabilities": {
            "streaming": False,
            "extensions": [
                {
                    "uri": UEOA_EXTENSION_URI,
                    "description": "UEOA registry metadata",
                    "params": {
                        "id": agent_id,
                        "owner": OWNER,
                        "kind": "agent",
                        "implementation": implementation,
                        "binding": "a2a",
                        "implements": [cap_id + "@" + (major or "1")],
                        "effects": "read",
                        "timeout_s": timeout_s,
                        "languages": ["en"],
                        "data_classification_ceiling": "confidential",
                    },
                }
            ],
        },
    }


def agent_argparser(default_port: int, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--run", metavar="INPUT", help="run once on a JSON file (or - for stdin) and print the result")
    return parser


def serve_or_run(card: dict[str, Any], capability: str, handler: Any, args: argparse.Namespace) -> None:
    if args.run:
        raw = sys.stdin.read() if args.run.strip() == "-" else Path(args.run).read_text(encoding="utf-8-sig")
        result = asyncio.run(handler(json.loads(raw), InvocationContext(capability, None)))
        print(json.dumps(result, indent=2))
        return
    base_url = "http" + "://" + args.host + ":" + str(args.port)
    app = create_agent_app(card, {capability: handler}, base_url)
    log.info("%s listening on %s", card["name"], base_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")