"""Example problems shipped with stratarena."""

from typing import Any

from stratarena.examples.binpacking import BinPacking
from stratarena.examples.gridbots import GridBots
from stratarena.problem import AnyProblem

PROBLEMS: dict[str, type[Any]] = {
    BinPacking.name: BinPacking,
    GridBots.name: GridBots,
}


def make_problem(name: str, options: dict[str, Any] | None = None) -> AnyProblem:
    """Instantiate a registered problem by name."""
    try:
        cls = PROBLEMS[name]
    except KeyError:
        known = ", ".join(sorted(PROBLEMS))
        raise ValueError(f"unknown problem {name!r} (known: {known})") from None
    problem: AnyProblem = cls(**(options or {}))
    return problem


__all__ = ["PROBLEMS", "BinPacking", "GridBots", "make_problem"]
