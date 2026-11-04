"""Run configuration, loaded from YAML and validated with pydantic."""

import hashlib
import json
from pathlib import Path
from typing import Any, Self

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
    arena: ArenaConfig = Field(default_factory=ArenaConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

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
