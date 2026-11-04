"""Run lifecycle: wake-ups, cancellation, shutdown, waiting TTL, global limits, timeouts and input redaction."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from ueoa.contracts import CapabilityContract, WorkflowSpec
from ueoa.gateway.service import InvocationRecord, Principal
from ueoa.orchestrator import runner as runner_module
from ueoa.orchestrator.runner import EXPIRED_DETAIL, RunController, RunError
from ueoa.orchestrator.scheduler import NodeState, RunStatus, eligible_nodes

pytestmark = pytest.mark.anyio

USER = Principal("tester", "user")
OTHER = Principal("other", "user")
THIRD = Principal("third", "user")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Gateway:
    """Gateway double: optional per-capability delays and gates, concurrency and cancellation counters."""

    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self.delays = delays or {}
        self.gates: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []
        self.active = 0
        self.peak = 0
        self.cancelled = 0

    def gate(self, capability: str) -> asyncio.Event:
        return self.gates.setdefault(capability, asyncio.Event())

    async def invoke(self, principal: Principal, capability_id: str, payload: dict[str, Any]) -> InvocationRecord:
        self.calls.append(capability_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if capability_id in self.gates:
                await self.gates[capability_id].wait()
            await asyncio.sleep(self.delays.get(capability_id, 0.01))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1
        return InvocationRecord(
            invocation_id=str(uuid.uuid4()),
            at="2026-01-01T00:00:00+00:00",
            principal=principal,
            capability=capability_id,
            status="succeeded",
            implementation=f"fake-{capability_id}@1.0.0",
            output={"flags": []} if capability_id == "result.score" else {"id": "x"},
        )


def _score(node_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "capability": "result.score",
        "inputs": {"results": [], "profile": "p"},
        "timeout_s": 30,
        **extra,
    }


def _ingest(node_id: str, **extra: Any) -> dict[str, Any]:
    return {"id": node_id, "capability": "item.ingest", "inputs": {"item": {}}, "timeout_s": 30, **extra}


ASK: dict[str, Any] = {
    "id": "ask",
    "type": "human_input",
    "input_schema": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}},
}


def _spec(*nodes: dict[str, Any], inputs: dict[str, Any] | None = None) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {"id": "life", "version": "1.0.0", "goal": "Lifecycle", "inputs": inputs or {}, "nodes": list(nodes)}
    )


def _fan_out(node_id: str = "each", **extra: Any) -> dict[str, Any]:
    return _ingest(node_id, for_each="$inputs.items", max_concurrency=8, inputs={"item": "$item"}, **extra)


ITEMS_INPUT: dict[str, Any] = {"items": {"type": "array"}}


def _controller(gateway: Gateway, capabilities: dict[str, CapabilityContract], **options: Any) -> RunController:
    return RunController(gateway, lambda: capabilities, max_backoff_s=0, **options)


async def _until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_node_timeout_raises_effective_timeout_and_is_shown(
    capabilities: dict[str, CapabilityContract],
) -> None:
    gateway = Gateway({"result.score": 1.3})
    controller = _controller(gateway, capabilities, node_timeout=lambda c: 3.0 if c == "result.score" else None)
    spec = _spec(_score("slow", timeout_s=1), _ingest("quick", timeout_s=5), ASK)

    run = await controller.execute(spec, {}, USER)

    assert run.nodes["slow"].state is NodeState.SUCCEEDED, run.nodes["slow"].error
    assert [run.nodes[n].timeout_s for n in ("slow", "quick", "ask")] == [3.0, 5, None]
    assert run.view()["nodes"]["slow"]["timeout_s"] == 3.0


async def test_submit_input_wakes_loop_while_other_branch_runs(capabilities: dict[str, CapabilityContract]) -> None:
    gateway = Gateway()
    release = gateway.gate("result.score")
    spec = _spec(_score("a"), ASK, _ingest("c", depends_on=["ask"], inputs={"item": "$nodes.ask.output"}))
    controller = _controller(gateway, capabilities)

    run = await controller.start(spec, {}, USER)
    await _until(
        lambda: run.nodes["ask"].state is NodeState.WAITING_INPUT and run.nodes["a"].state is NodeState.RUNNING
    )
    await controller.submit_input(run.id, "ask", {"id": "x"})
    await _until(lambda: run.nodes["c"].state is NodeState.SUCCEEDED)

    assert run.nodes["a"].state is NodeState.RUNNING
    release.set()
    run = await controller.wait(run.id, timeout=2)
    assert run.status is RunStatus.COMPLETED


async def test_cancel_stops_in_flight_work_and_is_idempotent(capabilities: dict[str, CapabilityContract]) -> None:
    gateway = Gateway()
    gateway.gate("result.score")
    gateway.gate("item.ingest")
    controller = _controller(gateway, capabilities)
    run = await controller.start(_spec(_score("a"), _fan_out(), ASK, inputs=ITEMS_INPUT), {"items": [{}, {}]}, USER)
    await _until(lambda: gateway.active == 3)

    cancelled = await controller.cancel(run.id)

    assert cancelled is run and run.status is RunStatus.CANCELLED and run.finished_at is not None
    assert {n: str(r.state) for n, r in run.nodes.items()} == {
        "a": "cancelled",
        "each": "cancelled",
        "ask": "cancelled",
    }
    assert [item.state for item in run.nodes["each"].items] == [NodeState.CANCELLED] * 2
    assert gateway.active == 0 and gateway.cancelled == 3
    assert run.events[-1].type == "run_cancelled" and run.events[-1].detail == "cancelled by request"
    events = len(run.events)
    assert await controller.cancel(run.id) is run and len(run.events) == events
    assert await controller.wait(run.id, timeout=1) is run

    gateway.gates.clear()
    done = await controller.execute(_spec(_ingest("x")), {}, USER)
    assert (await controller.cancel(done.id)).status is RunStatus.COMPLETED


async def test_drive_loop_error_cancels_child_tasks(
    capabilities: dict[str, CapabilityContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def flaky(spec: WorkflowSpec, states: Any) -> list[str]:
        calls.append(1)
        if len(calls) >= 3:
            raise RuntimeError("boom")
        return eligible_nodes(spec, states)

    monkeypatch.setattr(runner_module, "eligible_nodes", flaky)
    gateway = Gateway()
    gateway.gate("result.score")
    controller = _controller(gateway, capabilities)

    run = await controller.start(_spec(_score("slow"), _ingest("fast")), {}, USER)
    run = await controller.wait(run.id, timeout=2)

    assert run.status is RunStatus.FAILED and run.error == "RuntimeError: boom"
    assert run.nodes["fast"].state is NodeState.SUCCEEDED and run.nodes["slow"].state is NodeState.CANCELLED
    assert gateway.active == 0 and gateway.cancelled == 1


async def test_close_cancels_all_active_runs(capabilities: dict[str, CapabilityContract]) -> None:
    gateway = Gateway()
    gateway.gate("result.score")
    controller = _controller(gateway, capabilities)
    waiting = await controller.execute(_spec(ASK), {}, USER)
    busy = await controller.start(_spec(_score("a")), {}, USER)
    await _until(lambda: gateway.active == 1)

    await controller.close()

    assert [r.status for r in (waiting, busy)] == [RunStatus.CANCELLED] * 2
    assert {r.error for r in (waiting, busy)} == {"controller closed"}
    assert gateway.active == 0


async def test_waiting_runs_expire_and_can_be_evicted(capabilities: dict[str, CapabilityContract]) -> None:
    controller = _controller(Gateway(), capabilities, waiting_ttl_s=0.05, max_runs=1)
    run = await controller.execute(_spec(ASK), {}, USER)
    assert run.status is RunStatus.WAITING
    await asyncio.sleep(0.1)

    assert controller.get(run.id).status is RunStatus.CANCELLED
    assert run.nodes["ask"].state is NodeState.CANCELLED
    assert run.events[-1].type == "run_cancelled" and run.events[-1].detail == EXPIRED_DETAIL
    with pytest.raises(RunError, match="not waiting"):
        await controller.submit_input(run.id, "ask", {"id": "x"})

    second = await controller.execute(_spec(ASK), {}, USER)
    await asyncio.sleep(0.1)
    third = await controller.start(_spec(ASK), {}, USER)
    assert second.status is RunStatus.CANCELLED and controller.list() == [third]


async def test_active_run_limits(capabilities: dict[str, CapabilityContract]) -> None:
    controller = _controller(Gateway(), capabilities, max_active_runs=2, max_active_runs_per_principal=1)
    first = await controller.execute(_spec(ASK), {}, USER)

    with pytest.raises(RunError, match="too many active runs for 'tester'; the limit is 1"):
        await controller.start(_spec(ASK), {}, USER)
    await controller.execute(_spec(ASK), {}, OTHER)
    with pytest.raises(RunError, match="too many active runs; the limit is 2"):
        await controller.start(_spec(ASK), {}, THIRD)

    await controller.cancel(first.id)
    assert (await controller.execute(_spec(ASK), {}, USER)).status is RunStatus.WAITING


async def test_total_invocations_are_capped_across_runs(capabilities: dict[str, CapabilityContract]) -> None:
    gateway = Gateway({"item.ingest": 0.02})
    controller = _controller(gateway, capabilities, max_total_invocations=3)
    spec = _spec(_fan_out(), inputs=ITEMS_INPUT)

    runs = await asyncio.gather(*(controller.execute(spec, {"items": [{}] * 4}, USER) for _ in range(2)))

    assert [r.status for r in runs] == [RunStatus.COMPLETED] * 2
    assert gateway.calls.count("item.ingest") == 8 and gateway.peak == 3


async def test_view_can_redact_inputs_but_always_summarises_them(
    capabilities: dict[str, CapabilityContract],
) -> None:
    items = [{"id": "a"}, {"id": "b"}]
    run = await _controller(Gateway(), capabilities).execute(
        _spec(_fan_out(), inputs=ITEMS_INPUT), {"items": items}, USER
    )

    summary = {"items": len(json.dumps(items))}
    assert run.view()["inputs"] == {"items": items} and run.view()["inputs_summary"] == summary
    redacted = run.view(include_inputs=False)
    assert redacted["inputs"] is None and redacted["inputs_summary"] == summary


async def test_runner_timeout_cancels_invocation_and_records_error(
    capabilities: dict[str, CapabilityContract],
) -> None:
    gateway = Gateway({"result.score": 5, "item.ingest": 5})
    no_retry = {"timeout_s": 1, "retry": {"max_attempts": 1}}
    spec = _spec(_score("single", **no_retry), _fan_out(**no_retry), inputs=ITEMS_INPUT)

    run = await _controller(gateway, capabilities).execute(spec, {"items": [{}]}, USER)

    assert run.status is RunStatus.FAILED
    assert run.nodes["single"].error == "timeout: exceeded 1s"
    assert run.nodes["each"].items[0].error == "timeout: exceeded 1s"
    assert run.nodes["each"].error == "item 0: timeout: exceeded 1s"
    assert gateway.active == 0 and gateway.cancelled == 2
