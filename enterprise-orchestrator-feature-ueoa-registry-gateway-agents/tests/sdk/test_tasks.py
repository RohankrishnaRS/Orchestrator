"""ueoa_sdk.tasks: task-config agents (llm_call, analyst_reviewer, retrieve_classify) on neutral fixtures."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ueoa.registry.models import AgentCard
from ueoa_sdk.host import InvocationContext
from ueoa_sdk.retrieval import Index, chunk, locate, source_text
from ueoa_sdk.tasks import TaskDefinition, build_task_agent, load_task_file, render

pytestmark = pytest.mark.anyio

P1 = "".join(f"Battery note {i}: keep the battery pack dry and charge it at room temperature. " for i in range(7))
P2 = "".join(f"Charging note {i}: a full battery charge takes about two hours on the dock. " for i in range(7))
SECTIONS = [
    ("Battery", f"{P1.strip()}\n\n{P2.strip()}", 1),
    ("Cleaning", "Wipe the housing with a soft dry cloth. Never use solvents on the display.", 1),
    ("Storage", "Store the device in its case between uses. Remove the battery before storing it for a month.", 2),
]

TASKS_YAML = """
tasks:
  - name: topic_answer
    capability:
      id: topic.answer
      version: 1.0.0
      output_schema:
        type: object
        additionalProperties: false
        required: [answer]
        properties: {answer: {type: string}}
    execution_type: llm_call
    card: {id: topic-answerer, name: Topic Answerer, description: Answers about a topic.}
    system_prompt_template: "ANSWER about {{ topic }}."
    user_prompt_template: "Question: {{ question }}"

  - name: field_fill
    capability: {id: manual.fields.fill, version: 2.1.0}
    execution_type: analyst_reviewer
    max_rounds: 1
    retrieval: {document_from: document, items_from: fields, queries_from: [question], top_k: 3}
    output:
      items_key: rows
      copy_fields: [field]
      confidence_min: 60
      confidence_max: 100
    card: {id: field-filler, name: Field Filler, description: Fills fields from a manual.}
    system_prompt_template: "ANALYST for {{ document.document_id }}"
    user_prompt_template: |
      Fields: {{ items | tojson }}
      {% for chunk in chunks %}
      [{{ chunk.chunk_id }}] {{ chunk.text }}
      {% endfor %}
    reviewer_prompt_template: "REVIEWER for {{ items | length }} fields"

  - name: topic_find
    capability: {id: manual.topics.find, version: 1.0.0}
    execution_type: retrieve_classify
    retrieval: {document_from: document, items_from: topics, top_k: 4, expand_neighbors: 1}
    output: {results_key: found, target_field: topic, spans_field: passages}
    card: {id: topic-finder, name: Topic Finder, description: Finds topics in a manual.}
    knowledge:
      - name: battery
        aliases: [power pack]
        definition: "Charging and care of the battery."
        queries: [battery charge]
    fallback:
      definition: "Anything about {{ target }}."
      queries: ["{{ target }} instructions"]
    system_prompt_template: "CLASSIFY {{ target }}: {{ definition }}"
    user_prompt_template: |
      {% for chunk in chunks %}
      <{{ chunk.index }}> [{{ chunk.section }}] {{ chunk.text }}
      {% endfor %}
    validate_prompt_template: "VALIDATE {{ target }} on pages {{ pages }}"
"""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _document() -> tuple[dict[str, Any], str]:
    full, sections = "", []
    for n, (title, text, page) in enumerate(SECTIONS, 1):
        full += f"{title}\n"
        sections.append(
            {"section_id": f"s{n}", "title": title, "start": len(full), "end": len(full) + len(text), "text": text}
            | {"page": page}
        )
        full += f"{text}\n"
    return {"document_id": "manual", "status": "ok", "sections": sections}, full


class ScriptedLLM:
    def __init__(self, reply: Callable[[str, str], dict[str, Any]]) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str, *, chain_id: str | None = None) -> dict[str, Any]:
        self.calls.append((system, user))
        return self.reply(system, user)


@pytest.fixture
def tasks(tmp_path: Path) -> dict[str, TaskDefinition]:
    path = tmp_path / "tasks.yaml"
    path.write_text(TASKS_YAML, encoding="utf-8")
    return {t.name: t for t in load_task_file(path)}


async def _run(task: TaskDefinition, llm: ScriptedLLM, payload: dict[str, Any]) -> dict[str, Any]:
    _, handlers = build_task_agent(task, llm)
    return await handlers[task.capability.id](payload, InvocationContext(task.capability.id, "inv-1"))


# ---- retrieval -----------------------------------------------------------------------------------------------------


def test_chunks_have_absolute_offsets_and_search_dedups_queries() -> None:
    document, full = _document()
    chunks = chunk([document])
    assert [c.section for c in chunks] == ["Battery", "Battery", "Cleaning", "Storage"]
    for item in chunks:
        assert full[item.start : item.end] == item.text == source_text(document)[item.start : item.end]
    hits = Index([document]).search(["battery charge", "battery temperature", "battery"], top_k=10)
    assert len({c.chunk_id for c, _ in hits}) == len(hits) == 3
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert locate("keep the  battery\npack dry", P1) == (P1.index("keep"), P1.index("keep") + 25)
    assert locate("not there", P1) is None


# ---- llm_call ------------------------------------------------------------------------------------------------------


async def test_llm_call_renders_templates_and_appends_runtime_rules(tasks: dict[str, TaskDefinition]) -> None:
    llm = ScriptedLLM(lambda s, u: {"answer": "two hours"})
    output = await _run(tasks["topic_answer"], llm, {"topic": "batteries", "question": "How long to charge?"})
    assert output == {"answer": "two hours"}
    ((system, user),) = llm.calls
    assert system.startswith("ANSWER about batteries.")
    assert "untrusted data" in system and '"required": ["answer"]' in system
    assert user == "Question: How long to charge?"


async def test_llm_call_repairs_a_schema_violation_once(tasks: dict[str, TaskDefinition]) -> None:
    replies = iter([{"answer": 3}, {"answer": "fixed"}])
    llm = ScriptedLLM(lambda s, u: next(replies))
    output = await _run(tasks["topic_answer"], llm, {"topic": "t", "question": "q"})
    assert output == {"answer": "fixed"}
    assert len(llm.calls) == 2
    assert "did not match the required JSON schema (answer: 3 is not of type 'string')" in llm.calls[1][1]

    llm = ScriptedLLM(lambda s, u: {"other": True})
    with pytest.raises(ValueError, match="does not match the output schema"):
        await _run(tasks["topic_answer"], llm, {"topic": "t", "question": "q"})
    assert len(llm.calls) == 2


# ---- sandbox -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "{{ topic.__class__.__mro__ }}",
        "{{ topic.format.__globals__ }}",
        "{{ ''.__class__.__base__.__subclasses__() }}",
        "{{ cycler.__init__.__globals__.os.popen('echo hi').read() }}",
        "{{ topic.__init__.__globals__['__builtins__'] }}",
        "{{ missing_variable }}",
    ],
)
def test_sandbox_blocks_attribute_tricks_and_undefined_names(source: str) -> None:
    with pytest.raises(ValueError, match="prompt template failed"):
        render(source, {"topic": "batteries"})


async def test_unsafe_template_fails_before_the_model_is_called(tasks: dict[str, TaskDefinition]) -> None:
    task = tasks["topic_answer"].model_copy(
        update={"user_prompt_template": "{{ question.__class__.__subclasses__() }}"}
    )
    llm = ScriptedLLM(lambda s, u: {"answer": "x"})
    with pytest.raises(ValueError, match="prompt template failed"):
        await _run(task, llm, {"topic": "t", "question": "q"})
    assert llm.calls == []


def test_invalid_templates_and_incomplete_tasks_are_rejected(tasks: dict[str, TaskDefinition]) -> None:
    raw = tasks["topic_answer"].model_dump()
    with pytest.raises(ValidationError, match="invalid template"):
        TaskDefinition.model_validate(raw | {"system_prompt_template": "{% for x in %}"})
    with pytest.raises(ValidationError, match="needs `retrieval`"):
        TaskDefinition.model_validate(raw | {"execution_type": "retrieve_classify"})
    raw = tasks["field_fill"].model_dump()
    with pytest.raises(ValidationError, match="reviewer_prompt_template"):
        TaskDefinition.model_validate(raw | {"reviewer_prompt_template": None})


# ---- card ----------------------------------------------------------------------------------------------------------


def test_card_is_a_valid_ueoa_agent_card(tasks: dict[str, TaskDefinition]) -> None:
    raw, handlers = build_task_agent(tasks["field_fill"], ScriptedLLM(lambda s, u: {}))
    card = AgentCard.model_validate(raw)
    ext = card.ueoa
    assert (ext.id, ext.kind, ext.implementation, ext.binding) == ("field-filler", "agent", "llm_agent", "a2a")
    assert ext.implements == ["manual.fields.fill@2"]
    assert ext.effects.value == "read"
    assert [s.id for s in card.skills] == list(handlers) == ["manual.fields.fill"]


# ---- analyst_reviewer ----------------------------------------------------------------------------------------------

FIELDS = [
    {"field": "care", "question": "How should the battery pack be kept?"},
    {"field": "colour", "question": "What colour is the housing?"},
]


def _analyst(answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"items": [{"id": key} | value for key, value in answers.items()]}


async def test_analyst_reviewer_reasks_at_most_max_rounds_and_verifies_quotes(tasks: dict[str, TaskDefinition]) -> None:
    analyst = {
        "i1": {"value": "dry", "chunk_id": "c1", "quote": "keep the  battery\npack dry", "confidence": 120},
        "i2": {"value": "red", "chunk_id": "c3", "quote": "The housing is red.", "confidence": 90},
    }

    def reply(system: str, user: str) -> dict[str, Any]:
        if system.startswith("ANALYST"):
            return _analyst(analyst)
        return {"errors": [{"id": "i2", "issue": "colour not in the passages"}]}

    llm = ScriptedLLM(reply)
    document, full = _document()
    output = await _run(tasks["field_fill"], llm, {"document": document, "fields": FIELDS})

    stages = [system.split(" ")[0] for system, _ in llm.calls]
    assert stages == ["ANALYST", "REVIEWER", "ANALYST", "REVIEWER"]
    first_user = llm.calls[0][1]
    assert llm.calls[1][1].startswith(first_user) and "Analyst answer (JSON)" in llm.calls[1][1]
    assert "colour not in the passages" in llm.calls[2][1]
    assert all("untrusted data" in system for system, _ in llm.calls)

    care, colour = output["rows"]
    assert output["document_id"] == "manual"
    assert care == {
        "field": "care",
        "value": "dry",
        "page": "1",
        "confidence": 100.0,
        "quote": "keep the battery pack dry",
    }
    assert care["quote"] in full
    assert colour == {"field": "colour", "value": "", "page": "", "confidence": 60, "quote": None}


async def test_reviewer_approval_applies_corrections(tasks: dict[str, TaskDefinition]) -> None:
    analyst = {"i1": {"value": "damp", "chunk_id": "c1", "quote": "keep the battery pack dry", "confidence": 70}}

    def reply(system: str, user: str) -> dict[str, Any]:
        if system.startswith("ANALYST"):
            return _analyst(analyst)
        return {"approved": True, "items": [{"id": "i1", "value": "dry", "confidence": 95}]}

    llm = ScriptedLLM(reply)
    document, _ = _document()
    output = await _run(tasks["field_fill"], llm, {"document": document, "fields": FIELDS[:1]})
    assert len(llm.calls) == 2
    assert output["rows"] == [
        {"field": "care", "value": "dry", "page": "1", "confidence": 95.0, "quote": "keep the battery pack dry"}
    ]


# ---- retrieve_classify ---------------------------------------------------------------------------------------------


def _classifier(system: str, user: str) -> dict[str, Any]:
    if system.startswith("VALIDATE"):
        target = re.match(r"VALIDATE (\w[\w ]*?) on", system)
        valid = target is not None and target.group(1) != "solvent"
        return {"valid": valid, "confidence": "88.5", "summary": "Covers it." if valid else None}
    keyword = "battery" if "battery" in system else re.match(r"CLASSIFY (\w+)", system).group(1).lower()  # type: ignore[union-attr]
    flags = [
        {"index": int(index), "is_match": keyword in text.lower()}
        for index, text in re.findall(r"^<(\d+)> \[[^\]]*\] (.*)$", user, re.MULTILINE)
    ]
    return {"flags": [*flags, {"index": 99, "is_match": True}]}


async def test_retrieve_classify_stitches_flagged_chunks_verbatim(tasks: dict[str, TaskDefinition]) -> None:
    llm = ScriptedLLM(_classifier)
    document, full = _document()
    output = await _run(tasks["topic_find"], llm, {"document": document, "topics": ["Power Pack", "solvent", "warp"]})
    battery, solvent, warp = output["found"]

    assert battery["topic"] == "Power Pack" and battery["present"] is True
    assert (battery["summary"], battery["confidence"], battery["not_found_reason"]) == ("Covers it.", 88.5, None)
    stitched, storage = battery["passages"]
    assert (stitched["section"], stitched["page"], storage["section"], storage["page"]) == ("Battery", 1, "Storage", 2)
    assert stitched["quote"] == SECTIONS[0][1] and "\n\n" in stitched["quote"]
    for span in battery["passages"]:
        assert full[span["start"] : span["end"]] == span["quote"]

    assert (solvent["present"], solvent["passages"], solvent["confidence"]) == (False, [], 88.5)
    assert "validation rejected" in solvent["not_found_reason"]
    assert warp == {
        "topic": "warp",
        "present": False,
        "passages": [],
        "summary": None,
        "not_found_reason": "no matching passage retrieved",
        "confidence": None,
    }

    classify = [system for system, _ in llm.calls if system.startswith("CLASSIFY")]
    assert classify[0].startswith("CLASSIFY Power Pack: Charging and care of the battery.")
    assert classify[1].startswith("CLASSIFY solvent: Anything about solvent.")
    assert len(classify) == 2
    assert all("never include text" in system for system in classify)


async def test_retrieve_classify_reports_unreadable_documents(tasks: dict[str, TaskDefinition]) -> None:
    llm = ScriptedLLM(_classifier)
    document = {"document_id": "scan", "status": "unreadable", "sections": []}
    output = await _run(tasks["topic_find"], llm, {"document": document, "topics": ["battery"]})
    assert output["found"][0]["not_found_reason"] == "document unreadable"
    assert llm.calls == []
