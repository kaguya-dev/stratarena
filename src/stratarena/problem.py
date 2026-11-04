"""The interface a problem implements to be explored by stratarena."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from stratarena.strategy import ParamSpace, ParamStrategy

InstanceT = TypeVar("InstanceT")


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one strategy on one instance. Higher ``score`` is always better."""

    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)


class Problem(Protocol[InstanceT]):
    """A family of instances plus a way to score a strategy on one of them.

    Implementations must be picklable (they are sent to worker processes) and
    deterministic: the same seed yields the same instance, and the same strategy on the
    same instance yields the same result.
    """

    name: str

    def param_space(self) -> ParamSpace:
        """Parameters a ``ParamStrategy`` for this problem must provide."""
        ...

    def generate_instance(self, seed: int) -> InstanceT:
        """Build the instance identified by ``seed``."""
        ...

    def evaluate(self, strategy: ParamStrategy, instance: InstanceT) -> EvalResult:
        """Run ``strategy`` on ``instance``."""
        ...


@runtime_checkable
class GameProblem(Protocol[InstanceT]):
    """A two-player game: strategies are ranked by head-to-head results, not a score.

    ``play`` returns the points of the first player: 1 for a win, 0.5 for a draw and 0 for
    a loss. The arena plays every pair on both sides of each instance, so a first-player
    advantage cancels out. Same picklability and determinism rules as ``Problem``.
    """

    name: str

    def param_space(self) -> ParamSpace: ...

    def generate_instance(self, seed: int) -> InstanceT: ...

    def play(self, first: ParamStrategy, second: ParamStrategy, instance: InstanceT) -> float: ...


AnyProblem = Problem[Any] | GameProblem[Any]


@runtime_checkable
class HasBaselines(Protocol):
    """Optional: a problem can provide known heuristics to compare the search against."""

    def baselines(self) -> list[ParamStrategy]: ...
