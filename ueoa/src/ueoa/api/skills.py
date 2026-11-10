"""Reusable skill catalog backed by the platform document store."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ueoa.gateway.service import Principal
from ueoa.store import StoreError

KIND = "skills"
DEFAULT_SKILLS = (
    {
        "id": "document-creation",
        "name": "Document creation",
        "description": "Create structured documents from supplied information.",
        "instructions": "Organize supplied facts into a clear, audience-appropriate document. Do not invent facts.",
        "tags": ["document", "writing"],
    },
    {
        "id": "summarization",
        "name": "Summarization",
        "description": "Produce concise summaries that preserve key facts.",
        "instructions": "Summarize the supplied material accurately and retain important qualifications.",
        "tags": ["text", "analysis"],
    },
    {
        "id": "research",
        "name": "Research",
        "description": "Gather and synthesize relevant evidence from available sources.",
        "instructions": "Use available sources, distinguish evidence from inference, and identify uncertainty.",
        "tags": ["research", "analysis"],
    },
    {
        "id": "data-analysis",
        "name": "Data analysis",
        "description": "Analyze supplied data and explain the results.",
        "instructions": "Analyze only the supplied data, show relevant assumptions, and explain findings.",
        "tags": ["data", "analysis"],
    },
    {
        "id": "information-extraction",
        "name": "Information extraction",
        "description": "Extract requested facts into a structured representation.",
        "instructions": "Extract only information supported by the source and preserve requested structure.",
        "tags": ["extraction", "structured-data"],
    },
    {
        "id": "validation",
        "name": "Validation",
        "description": "Check content against explicit requirements and report issues.",
        "instructions": "Check each stated requirement, cite the relevant evidence, and report any failure.",
        "tags": ["quality", "review"],
    },
    {
        "id": "content-generation",
        "name": "Content generation",
        "description": "Generate original content from an approved brief and context.",
        "instructions": "Follow the brief, use the provided context, and do not fabricate unsupported claims.",
        "tags": ["content", "writing"],
    },
)


class SkillRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=2000)
    instructions: str = Field(min_length=1, max_length=10000)
    tags: list[str] = Field(default_factory=list, max_length=30)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


async def on_startup(services: Any, _settings: Any) -> None:
    store = services.store
    builtin_ids = {item["id"] for item in DEFAULT_SKILLS}
    for existing in store.list(KIND):
        if existing.get("builtin") and existing.get("id") not in builtin_ids:
            store.delete(KIND, existing["id"])
    for skill in DEFAULT_SKILLS:
        store.put(
            KIND,
            skill["id"],
            {
                **skill,
                "builtin": True,
                "owner": "platform",
                "created_at": _now(),
                "updated_at": _now(),
            },
        )


def create_router(
    services: Callable[[], Any],
    principal: Callable[..., Principal],
    admin: Callable[..., Principal],
    _settings: Any,
) -> APIRouter:
    router = APIRouter()
    Caller = Annotated[Principal, Depends(principal)]
    Admin = Annotated[Principal, Depends(admin)]

    @router.get("/api/v1/skills")
    def list_skills(_: Caller) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return sorted(
            services().store.list(KIND),
            key=lambda item: (not item.get("builtin", False), item["name"].casefold()),
        )

    @router.post("/api/v1/skills", status_code=201)
    def create_skill(body: SkillRequest, who: Admin) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        store = services().store
        skill_id = body.name.strip().lower().replace(" ", "-")
        try:
            if store.get(KIND, skill_id) is not None:
                raise HTTPException(409, f"skill '{skill_id}' already exists")
            skill = {
                "id": skill_id,
                **body.model_dump(),
                "builtin": False,
                "owner": who.name,
                "created_at": _now(),
                "updated_at": _now(),
            }
            store.put(KIND, skill_id, skill)
            return skill
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.put("/api/v1/skills/{skill_id}")
    def update_skill(skill_id: str, body: SkillRequest, who: Admin) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        store = services().store
        try:
            existing = store.get(KIND, skill_id)
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc
        if existing is None:
            raise HTTPException(404, f"unknown skill '{skill_id}'")
        if existing.get("builtin"):
            raise HTTPException(409, "default skills are read-only")
        if existing.get("owner") != who.name and who.role != "admin":
            raise HTTPException(403, f"skill '{skill_id}' belongs to another user")
        skill = {**existing, **body.model_dump(), "updated_at": _now()}
        store.put(KIND, skill_id, skill)
        return skill

    @router.delete("/api/v1/skills/{skill_id}")
    def delete_skill(skill_id: str, who: Admin) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        store = services().store
        try:
            existing = store.get(KIND, skill_id)
        except StoreError as exc:
            raise HTTPException(422, str(exc)) from exc
        if existing is None:
            raise HTTPException(404, f"unknown skill '{skill_id}'")
        if existing.get("builtin"):
            raise HTTPException(409, "default skills cannot be deleted")
        if existing.get("owner") != who.name and who.role != "admin":
            raise HTTPException(403, f"skill '{skill_id}' belongs to another user")
        store.delete(KIND, skill_id)
        return {"id": skill_id, "deleted": True}

    return router
