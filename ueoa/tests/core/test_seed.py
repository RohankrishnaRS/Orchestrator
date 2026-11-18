from ueoa.contracts import CapabilityContract, Effect, WorkflowSpec


def test_fixture_catalog_loads(capabilities: dict[str, CapabilityContract]) -> None:
    assert "item.analyze" in capabilities
    assert capabilities["action.apply"].effects is Effect.WRITE
    assert all(c.effects is Effect.READ for k, c in capabilities.items() if k != "action.apply")


def test_pipeline_loads(pipeline: WorkflowSpec) -> None:
    assert pipeline.node("apply").effect is Effect.WRITE
    assert pipeline.node("choose").type == "human_input"
