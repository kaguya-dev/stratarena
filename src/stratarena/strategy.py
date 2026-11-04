"""Strategies: the candidates that compete in the arena."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """One continuous parameter with inclusive bounds."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low ({self.low}) must be < high ({self.high})")


ParamSpace = tuple[ParamSpec, ...]


def clip_to_space(space: ParamSpace, values: Sequence[float]) -> tuple[float, ...]:
    """Clip ``values`` into the bounds of ``space``."""
    if len(values) != len(space):
        raise ValueError(f"expected {len(space)} values, got {len(values)}")
    return tuple(min(max(float(v), p.low), p.high) for v, p in zip(values, space, strict=True))


@dataclass(frozen=True)
class ParamStrategy:
    """A strategy described by a vector of bounded parameters.

    The problem decides what the parameters mean; ``parents`` records lineage once
    hybridization creates offspring.
    """

    id: str
    values: tuple[float, ...]
    space: ParamSpace
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.values) != len(self.space):
            raise ValueError(
                f"{self.id}: expected {len(self.space)} values, got {len(self.values)}"
            )
        for value, spec in zip(self.values, self.space, strict=True):
            if not spec.low <= value <= spec.high:
                raise ValueError(
                    f"{self.id}: {spec.name}={value} outside [{spec.low}, {spec.high}]"
                )

    @classmethod
    def random(cls, id: str, space: ParamSpace, rng: np.random.Generator) -> "ParamStrategy":
        """Sample a strategy uniformly inside ``space``."""
        values = tuple(float(rng.uniform(p.low, p.high)) for p in space)
        return cls(id=id, values=values, space=space)

    @classmethod
    def from_mapping(
        cls, id: str, space: ParamSpace, params: Mapping[str, float]
    ) -> "ParamStrategy":
        """Build a strategy from ``{name: value}``; every parameter must be given."""
        missing = {p.name for p in space} - params.keys()
        if missing:
            raise ValueError(f"{id}: missing parameters {sorted(missing)}")
        return cls(id=id, values=tuple(float(params[p.name]) for p in space), space=space)

    def as_dict(self) -> dict[str, float]:
        return {p.name: v for p, v in zip(self.space, self.values, strict=True)}

    def __getitem__(self, name: str) -> float:
        for spec, value in zip(self.space, self.values, strict=True):
            if spec.name == name:
                return value
        raise KeyError(name)
