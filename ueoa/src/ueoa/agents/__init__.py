"""Agents the platform hosts itself, defined declaratively (agent studio)."""

from .declarative import (
    HOSTED_SECRET_REF,
    AgentDefinition,
    CapabilitySpec,
    FieldSpec,
    agent_card,
    build_handler,
    capability_contract,
)

__all__ = [
    "HOSTED_SECRET_REF",
    "AgentDefinition",
    "CapabilitySpec",
    "FieldSpec",
    "agent_card",
    "build_handler",
    "capability_contract",
]
