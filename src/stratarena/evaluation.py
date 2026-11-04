"""Scoring strategies on instances, sequentially or on a process pool."""

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from types import TracebackType
from typing import Any, Self

from stratarena.problem import Problem
from stratarena.strategy import ParamStrategy


def evaluate_on_instance(
    problem: Problem[Any], strategies: Sequence[ParamStrategy], seed: int
) -> list[float]:
    """Score every strategy on the instance ``seed``. Runs inside worker processes."""
    instance = problem.generate_instance(seed)
    return [problem.evaluate(s, instance).score for s in strategies]


class Evaluator:
    """Evaluates strategies, one task per instance.

    With ``workers > 1`` it must be used as a context manager, which owns a ``spawn``
    process pool for its lifetime. Results come back in instance order either way, so the
    number of workers never changes the output.
    """

    def __init__(self, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        if self.workers > 1:
            ctx = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def scores(
        self, problem: Problem[Any], strategies: Sequence[ParamStrategy], seeds: Sequence[int]
    ) -> dict[str, list[float]]:
        """Return ``{strategy_id: [score on seeds[0], score on seeds[1], ...]}``."""
        task = partial(evaluate_on_instance, problem, list(strategies))
        if self.workers > 1 and self._pool is None:
            raise RuntimeError("use Evaluator as a context manager when workers > 1")
        rows = list(self._pool.map(task, seeds) if self._pool else map(task, seeds))
        return {s.id: [row[j] for row in rows] for j, s in enumerate(strategies)}
