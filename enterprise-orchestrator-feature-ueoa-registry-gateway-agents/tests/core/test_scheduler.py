from collections.abc import Callable

from hypothesis import given, settings
from hypothesis import strategies as st

from ueoa.contracts import WorkflowSpec
from ueoa.orchestrator.graph import topological_levels, topological_order
from ueoa.orchestrator.scheduler import (
    NodeState,
    RunStatus,
    aggregate_item_states,
    blocked_nodes,
    eligible_nodes,
    initial_states,
    must_skip,
    nodes_to_rerun,
    nodes_to_resume,
    run_status,
)


def _drive(
    spec: WorkflowSpec,
    states: dict[str, NodeState],
    condition: Callable[[str], bool],
    outcome: Callable[[str], NodeState] = lambda _node_id: NodeState.SUCCEEDED,
) -> list[list[str]]:
    """Simulate the controller loop; returns the ready waves for inspection."""
    waves: list[list[str]] = []
    while eligible := eligible_nodes(spec, states):
        wave: list[str] = []
        for node_id in eligible:
            if must_skip(spec, node_id, states) or not condition(node_id):
                states[node_id] = NodeState.SKIPPED
            else:
                states[node_id] = outcome(node_id)
                wave.append(node_id)
        waves.append(wave)
    return waves


def test_pipeline_levels(pipeline: WorkflowSpec) -> None:
    assert topological_levels(pipeline) == [
        ["ingest"],
        ["analyze"],
        ["check"],
        ["score", "contrast"],
        ["merge"],
        ["summarize"],
        ["choose"],
        ["plan"],
        ["apply"],
    ]


def test_full_run_with_optional_branch_and_actions(pipeline: WorkflowSpec) -> None:
    states = initial_states(pipeline)
    _drive(pipeline, states, condition=lambda _n: True)
    assert all(state is NodeState.SUCCEEDED for state in states.values())
    assert run_status(pipeline, states) is RunStatus.COMPLETED


def test_skipped_optional_branch_does_not_block_join(pipeline: WorkflowSpec) -> None:
    states = initial_states(pipeline)
    _drive(pipeline, states, condition=lambda node_id: node_id != "contrast")
    assert states["contrast"] is NodeState.SKIPPED
    assert states["merge"] is NodeState.SUCCEEDED  # guarded by | default(null)


def test_no_selection_skips_plan_and_propagates_to_apply(pipeline: WorkflowSpec) -> None:
    states = initial_states(pipeline)
    _drive(pipeline, states, condition=lambda node_id: node_id != "plan")
    assert states["plan"] is NodeState.SKIPPED
    assert states["apply"] is NodeState.SKIPPED  # for_each references plan unguarded
    assert run_status(pipeline, states) is RunStatus.COMPLETED


def test_waiting_for_input_and_approval(pipeline: WorkflowSpec) -> None:
    states = initial_states(pipeline)

    def outcome(node_id: str) -> NodeState:
        return NodeState.WAITING_INPUT if node_id == "choose" else NodeState.SUCCEEDED

    _drive(pipeline, states, condition=lambda _n: True, outcome=outcome)
    assert states["choose"] is NodeState.WAITING_INPUT
    assert run_status(pipeline, states) is RunStatus.WAITING
    assert states["plan"] is NodeState.PENDING


def test_failure_blocks_downstream_and_resume_targets_only_failed(pipeline: WorkflowSpec) -> None:
    states = initial_states(pipeline)

    def outcome(node_id: str) -> NodeState:
        return NodeState.FAILED if node_id == "analyze" else NodeState.SUCCEEDED

    _drive(pipeline, states, condition=lambda _n: True, outcome=outcome)
    assert states["ingest"] is NodeState.SUCCEEDED
    assert run_status(pipeline, states) is RunStatus.FAILED
    assert "check" in blocked_nodes(pipeline, states)
    assert nodes_to_resume(states) == ["analyze"]

    states["analyze"] = NodeState.PENDING
    _drive(pipeline, states, condition=lambda _n: True)
    assert run_status(pipeline, states) is RunStatus.COMPLETED


def test_rejected_write_completes_as_rejected(pipeline: WorkflowSpec) -> None:
    states = initial_states(pipeline)

    def outcome(node_id: str) -> NodeState:
        return NodeState.REJECTED if node_id == "apply" else NodeState.SUCCEEDED

    _drive(pipeline, states, condition=lambda _n: True, outcome=outcome)
    assert run_status(pipeline, states) is RunStatus.COMPLETED_REJECTED


def test_refinement_reruns_downstream_only(pipeline: WorkflowSpec) -> None:
    assert nodes_to_rerun(pipeline, {"score"}) == {"score", "merge", "summarize", "choose", "plan", "apply"}


def _chain(reference: str) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "id": "w",
            "version": "1.0.0",
            "goal": "g",
            "nodes": [
                {"id": "a", "capability": "test.cap", "timeout_s": 10},
                {"id": "b", "capability": "test.cap", "depends_on": ["a"], "timeout_s": 10},
                {"id": "c", "capability": "test.cap", "depends_on": ["b"], "inputs": {"x": reference}, "timeout_s": 10},
            ],
        }
    )


def test_skip_propagates_to_nodes_referencing_a_skipped_ancestor() -> None:
    states = {"a": NodeState.SKIPPED, "b": NodeState.SUCCEEDED, "c": NodeState.PENDING}
    assert must_skip(_chain("$nodes.a.output"), "c", states)
    assert not must_skip(_chain("$nodes.a.output | default(null)"), "c", states)
    assert not must_skip(_chain("$nodes.b.output"), "c", states)
    assert not must_skip(_chain("$nodes.a.output"), "b", states)


def test_aggregate_item_states() -> None:
    s = NodeState
    assert aggregate_item_states([]) is s.SUCCEEDED
    assert aggregate_item_states([s.SUCCEEDED, s.RUNNING]) is s.RUNNING
    assert aggregate_item_states([s.SUCCEEDED, s.FAILED]) is s.FAILED
    assert aggregate_item_states([s.SUCCEEDED, s.WAITING_APPROVAL]) is s.WAITING_APPROVAL
    assert aggregate_item_states([s.SUCCEEDED, s.REJECTED]) is s.SUCCEEDED
    assert aggregate_item_states([s.REJECTED, s.REJECTED]) is s.REJECTED


@st.composite
def random_dags(draw: st.DrawFn) -> WorkflowSpec:
    size = draw(st.integers(min_value=1, max_value=12))
    nodes = []
    for index in range(size):
        deps = draw(st.lists(st.integers(min_value=0, max_value=max(index - 1, 0)), max_size=3, unique=True))
        nodes.append(
            {
                "id": f"n{index}",
                "capability": "test.cap",
                "depends_on": [f"n{d}" for d in deps if d < index],
                "timeout_s": 10,
            }
        )
    return WorkflowSpec.model_validate({"id": "w", "version": "1.0.0", "goal": "g", "nodes": nodes})


@settings(max_examples=200, deadline=None)
@given(spec=random_dags(), skip_mask=st.lists(st.booleans(), min_size=12, max_size=12))
def test_scheduler_properties(spec: WorkflowSpec, skip_mask: list[bool]) -> None:
    order = topological_order(spec)
    assert sorted(order) == sorted(node.id for node in spec.nodes)
    states = initial_states(spec)
    executed: list[str] = []
    while eligible := eligible_nodes(spec, states):
        for node_id in eligible:
            node = spec.node(node_id)
            assert all(states[dep] in {NodeState.SUCCEEDED, NodeState.SKIPPED} for dep in node.depends_on)
            if skip_mask[int(node_id[1:])]:
                states[node_id] = NodeState.SKIPPED
            else:
                states[node_id] = NodeState.SUCCEEDED
                executed.append(node_id)
    assert all(state in {NodeState.SUCCEEDED, NodeState.SKIPPED} for state in states.values())
    assert run_status(spec, states) is RunStatus.COMPLETED
    position = {node_id: index for index, node_id in enumerate(executed)}
    for node_id in executed:
        for dep in spec.node(node_id).depends_on:
            if dep in position:
                assert position[dep] < position[node_id]
