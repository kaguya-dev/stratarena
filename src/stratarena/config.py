"""Run configuration, loaded from YAML and validated with pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArenaConfig(BaseModel):
    """How a battle royale is run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instances_per_round: int = Field(default=8, ge=1)
    elimination_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    finalists: int = Field(default=4, ge=1)
    max_rounds: int = Field(default=10, ge=1)
    workers: int = Field(default=1, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=100)


class RunConfig(BaseModel):
    """Everything needed to reproduce a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=0, ge=0)
    problem: str
    problem_options: dict[str, Any] = Field(default_factory=dict)
    population: int = Field(default=16, ge=2)
    arena: ArenaConfig = Field(default_factory=ArenaConfig)


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML run configuration."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig.model_validate(data)
