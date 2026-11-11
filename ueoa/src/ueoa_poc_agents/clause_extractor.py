# pyright: basic
"""Agent 1 - Clause Extractor: finds requested clause types in a contract and returns the exact clause text.

Standalone:  python -m ueoa_poc_agents.clause_extractor [--port 8201]
One-shot:    python -m ueoa_poc_agents.clause_extractor --run payload.json
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ueoa.a2a import InvocationContext
from ueoa_poc_agents import PORTS
from ueoa_poc_agents._base import agent_argparser, load_document_text, poc_card, serve_or_run

log = logging.getLogger(__name__)

AGENT_ID = "poc-clause-extractor"
CAPABILITY = "poc.clause.extract"
DEFAULT_CLAUSE_TYPES = ["confidentiality", "indemnification", "limitation_of_liability"]
CLAUSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "confidentiality": ("confidential", "non-disclosure", "nondisclosure", "proprietary information"),
    "indemnification": ("indemnif", "hold harmless", "defend and hold"),
    "limitation_of_liability": ("limitation of liability", "limitation on liability", "liability", "liable"),
    "termination": ("terminat",),
    "governing_law": ("governing law", "governed by", "jurisdiction"),
    "payment_terms": ("payment", "fees", "invoice"),
    "data_protection": ("data protection", "personal data", "gdpr", "privacy"),
    "insurance": ("insurance",),
}
_HEADING_RE = re.compile(
    r"^\s*(?:(?:article|section|clause)\s+)?(?:\d+(?:\.\d+)*|[ivxlc]+)[.)]?\s+\S.{0,80}$", re.IGNORECASE
)

CARD = poc_card(
    agent_id=AGENT_ID,
    name="Clause Extractor",
    description="Finds named clause types in a contract and returns the verbatim clause text with its location.",
    capability=f"{CAPABILITY}@1",
    skill_description="Given a contract and clause types, returns per type whether it is present and the exact text.",
    examples=["Does this NDA contain a confidentiality clause?", "Extract the indemnification clause"],
    tags=["contracts", "legal", "extraction"],
)


def normalise_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def split_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        para, start, end = match.group(0), match.start(), match.end()
        first = para.strip().splitlines()[0]
        heading = bool(_HEADING_RE.match(first)) or (len(first) <= 60 and first.isupper())
        if heading or current is None:
            current = {"heading": first.strip() if heading else "", "start": start, "end": end, "text": para}
            sections.append(current)
        else:
            current["end"] = end
            current["text"] = text[current["start"] : end]
    return sections


def extract(text: str, clause_types: list[str]) -> list[dict[str, Any]]:
    sections = split_sections(text)
    results: list[dict[str, Any]] = []
    for clause_type in clause_types:
        fallback = tuple(p for p in clause_type.split("_") if len(p) > 3) or (clause_type,)
        words = CLAUSE_KEYWORDS.get(clause_type) or fallback
        best, best_score = None, 0
        for section in sections:
            heading, body = section["heading"].lower(), section["text"].lower()
            score = sum(3 for w in words if w in heading) + sum(body.count(w) for w in words)
            if score > best_score:
                best, best_score = section, score
        if best is None or best_score == 0:
            results.append(
                {"clause_type": clause_type, "found": False, "heading": "", "text": "", "verified": False,
                 "confidence": 0.0, "notes": "no section mentions this clause type"}
            )
            continue
        results.append(
            {"clause_type": clause_type, "found": True, "heading": best["heading"], "text": best["text"].strip(),
             "start": best["start"], "end": best["end"], "verified": True,
             "confidence": min(0.95, 0.5 + 0.1 * best_score), "notes": f"keyword match ({best_score})"}
        )
    return results


async def handle(payload: dict[str, Any], ctx: InvocationContext) -> dict[str, Any]:
    text = load_document_text(payload)
    requested = payload.get("clause_types") or DEFAULT_CLAUSE_TYPES
    if not isinstance(requested, list) or not all(isinstance(t, str) and t for t in requested):
        raise ValueError("'clause_types' must be a non-empty list of strings")
    clause_types = [normalise_type(t) for t in requested]
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    document_id = str(document.get("document_id") or document.get("sample") or payload.get("document_id") or "document")
    clauses = extract(text, clause_types)
    found = [c["clause_type"] for c in clauses if c["found"]]
    missing = [c["clause_type"] for c in clauses if not c["found"]]
    log.info("%s: %s -> found=%s missing=%s", AGENT_ID, document_id, found, missing)
    summary = f"Found {len(found)} of {len(clause_types)} requested clause types"
    summary += f"; missing: {', '.join(missing)}." if missing else "."
    return {"document_id": document_id, "clauses": clauses, "found_count": len(found), "missing": missing,
            "summary": summary, "mode": "offline"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = agent_argparser(PORTS[AGENT_ID], "POC Agent 1 - Clause Extractor").parse_args()
    serve_or_run(CARD, CAPABILITY, handle, args)


if __name__ == "__main__":
    main()