"""Load capability contracts and workflow specs from YAML files (seed packs are supplied by the deployment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ueoa.contracts import CapabilityContract, WorkflowSpec


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_capabilities(directory: Path) -> dict[str, CapabilityContract]:
    catalog: dict[str, CapabilityContract] = {}
    for path in sorted(directory.glob("*.yaml")):
        documents = _load_yaml(path) or []
        for raw in documents:
            capability = CapabilityContract.model_validate(raw)
            if capability.id in catalog:
                raise ValueError(f"{path.name}: capability '{capability.id}' defined twice")
            catalog[capability.id] = capability
    return catalog


def load_workflow(path: Path) -> WorkflowSpec:
    return WorkflowSpec.model_validate(_load_yaml(path))
