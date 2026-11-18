from pathlib import Path

import pytest

from ueoa.contracts import CapabilityContract, WorkflowSpec
from ueoa.registry.seed import load_capabilities, load_workflow

FIXTURE_CATALOG = Path(__file__).resolve().parent / "fixtures" / "catalog"


@pytest.fixture(scope="session")
def capabilities() -> dict[str, CapabilityContract]:
    return load_capabilities(FIXTURE_CATALOG / "capabilities")


@pytest.fixture
def pipeline() -> WorkflowSpec:
    return load_workflow(FIXTURE_CATALOG / "workflows" / "pipeline.yaml")
