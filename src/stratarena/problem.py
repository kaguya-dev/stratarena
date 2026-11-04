"""The interface a problem implements to be explored by stratarena."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

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
