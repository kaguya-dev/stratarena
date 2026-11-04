"""Agents that propose new candidates between generations."""

from stratarena.agents.base import AgentContext, AgentEvent, Proposal, smoke_test
from stratarena.agents.critic import CriticAgent
from stratarena.agents.generator import CodeGeneratorAgent, extract_code
from stratarena.agents.orchestrator import Orchestrator
from stratarena.agents.tuner import ParamTunerAgent

__all__ = [
    "AgentContext",
    "AgentEvent",
    "CodeGeneratorAgent",
    "CriticAgent",
    "Orchestrator",
    "ParamTunerAgent",
    "Proposal",
    "extract_code",
    "smoke_test",
]
