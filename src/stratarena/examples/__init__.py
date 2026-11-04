"""Example problems shipped with stratarena."""

from typing import Any

from stratarena.examples.binpacking import BinPacking
from stratarena.problem import Problem

PROBLEMS: dict[str, type[Any]] = {
    BinPacking.name: BinPacking,
}


def make_problem(name: str, options: dict[str, Any] | None = None) -> Problem[Any]:
    """Instantiate a registered problem by name."""
    try:
        cls = PROBLEMS[name]
    except KeyError:
        known = ", ".join(sorted(PROBLEMS))
        raise ValueError(f"unknown problem {name!r} (known: {known})") from None
    problem: Problem[Any] = cls(**(options or {}))
    return problem


__all__ = ["PROBLEMS", "BinPacking", "make_problem"]
