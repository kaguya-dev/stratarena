"""Run configuration, loaded from YAML and validated with pydantic."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArenaConfig(BaseModel):
    """How a battle royale is run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instances_per_round: int = Field(default=8, ge=1)
    elimination_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    finalists: int = Field(default=4, ge=1)
    max_rounds: int = Field(default=10, ge=1)
    workers: int = Field(default=1, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=100)


class HybridConfig(BaseModel):
    """How finalists are recombined into the next generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crossover_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    mutation_scale: float = Field(default=0.1, gt=0.0, le=1.0)
    immigrants: int = Field(default=2, ge=0)


class SandboxConfig(BaseModel):
    """Limits for code strategies (see ``stratarena.sandbox``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_timeout_s: float = Field(default=2.0, gt=0.0)
    cpu_seconds: int = Field(default=30, ge=1)
    memory_mb: int = Field(default=256, ge=64)


class LLMConfig(BaseModel):
    """Which language model the agents use. ``none`` runs without any model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["none", "ollama", "anthropic"] = "none"
    model: str | None = None
    """Defaults: ``llama3.2`` for Ollama, ``claude-opus-5-5`` for Anthropic."""
    host: str | None = None
    """Ollama server; defaults to ``OLLAMA_HOST`` or ``http://localhost:11434``."""
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    """Ollama only."""
    seed: int = Field(default=0, ge=0)
    """Ollama only."""
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    """Anthropic only."""
    cache_dir: str = ".stratarena-cache/llm"
    """Every answer is stored here, keyed by a hash of the request, so runs replay."""


class AgentsConfig(BaseModel):
    """Agents that propose candidates between generations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    code_candidates: int = Field(default=2, ge=0)
    """Code strategies the generator writes per generation (needs an LLM)."""
    max_attempts: int = Field(default=2, ge=1)
    """Tries per code candidate; failed attempts are retried with the error as feedback."""
    critic: bool = True
    """Review accepted code candidates and feed the notes to the next generation."""
    tuner_candidates: int = Field(default=2, ge=0)
    """Parametric candidates from local search around the best finalist (no LLM)."""
    tuner_iterations: int = Field(default=10, ge=1)
    tuner_instances: int = Field(default=6, ge=1)
    smoke_instances: int = Field(default=2, ge=1)
    """Instances a code candidate must handle without errors before it may compete."""
    max_parallel: int = Field(default=4, ge=1)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class RunConfig(BaseModel):
    """Everything needed to reproduce a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=0, ge=0)
    problem: str
    problem_options: dict[str, Any] = Field(default_factory=dict)
    population: int = Field(default=16, ge=2)
    generations: int = Field(default=5, ge=1)
    holdout_instances: int = Field(default=50, ge=2)
    tracking_instances: int = Field(default=30, ge=1)
    baselines_compete: bool = False
    sparring: bool = False
    """Games only: the baselines play in every arena round robin as fixed opponents that
    are never ranked or eliminated. They then also stop being unseen opponents in the
    held-out validation (the maps still are unseen)."""
    arena: ArenaConfig = Field(default_factory=ArenaConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)

    @model_validator(mode="after")
    def _sparring_or_compete(self) -> Self:
        if self.sparring and self.baselines_compete:
            raise ValueError("baselines can either compete or spar, not both")
        return self

    @model_validator(mode="after")
    def _room_for_offspring(self) -> Self:
        if (
            self.generations > 1
            and self.population <= self.arena.finalists + self.hybrid.immigrants
        ):
            raise ValueError("population must exceed arena.finalists + hybrid.immigrants")
        return self

    def fingerprint(self) -> str:
        """Short stable hash of the configuration, used to name run directories."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML run configuration."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig.model_validate(data)
