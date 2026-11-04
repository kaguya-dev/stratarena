"""Scoring strategies on instances, sequentially or on a process pool."""

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from types import TracebackType
from typing import Any, Self

from stratarena.problem import AnyProblem, GameProblem
from stratarena.rating import PairStats
from stratarena.strategy import Strategy

# Per instance: one score per strategy, plus (a, b, points of a, games) for game problems.
InstanceOutcome = tuple[list[float], list[tuple[str, str, float, int]]]


@dataclass(frozen=True)
class Evaluation:
    scores: dict[str, list[float]]
    """``{strategy_id: [score on seeds[0], score on seeds[1], ...]}``."""
    pairs: PairStats = field(default_factory=PairStats)
    """Head-to-head results (game problems in round-robin mode only)."""


def _both_sides(problem: GameProblem[Any], a: Strategy, b: Strategy, inst: Any) -> float:
    """Points of ``a`` over two games against ``b``, one as each player."""
    return problem.play(a, b, inst) + (1.0 - problem.play(b, a, inst))


def evaluate_on_instance(
    problem: AnyProblem,
    strategies: Sequence[Strategy],
    opponents: Sequence[Strategy] | None,
    seed: int,
) -> InstanceOutcome:
    """Score every strategy on the instance ``seed``. Runs inside worker processes.

    For a game, the score is the mean points per game: against every other strategy in
    ``strategies`` (round robin), or against a fixed set of ``opponents`` when given.
    """
    instance = problem.generate_instance(seed)
    if not isinstance(problem, GameProblem):
        return [problem.evaluate(s, instance).score for s in strategies], []

    if opponents is not None:
        scores = []
        for s in strategies:
            rivals = [o for o in opponents if o.id != s.id]
            pts = sum(_both_sides(problem, s, o, instance) for o in rivals)
            scores.append(pts / (2 * len(rivals)) if rivals else 0.5)
        return scores, []

    n = len(strategies)
    points = [0.0] * n
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pts = _both_sides(problem, strategies[i], strategies[j], instance)
            points[i] += pts
            points[j] += 2.0 - pts
            pairs.append((strategies[i].id, strategies[j].id, pts, 2))
    return [p / (2 * (n - 1)) if n > 1 else 0.5 for p in points], pairs


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

    def evaluate(
        self,
        problem: AnyProblem,
        strategies: Sequence[Strategy],
        seeds: Sequence[int],
        opponents: Sequence[Strategy] | None = None,
    ) -> Evaluation:
        task = partial(
            evaluate_on_instance,
            problem,
            list(strategies),
            None if opponents is None else list(opponents),
        )
        if self.workers > 1 and self._pool is None:
            raise RuntimeError("use Evaluator as a context manager when workers > 1")
        rows = list(self._pool.map(task, seeds) if self._pool else map(task, seeds))
        pairs = PairStats()
        for _, instance_pairs in rows:
            for a, b, pts, games in instance_pairs:
                pairs.record(a, b, pts, games)
        scores = {s.id: [row[0][j] for row in rows] for j, s in enumerate(strategies)}
        return Evaluation(scores, pairs)

    def scores(
        self,
        problem: AnyProblem,
        strategies: Sequence[Strategy],
        seeds: Sequence[int],
        opponents: Sequence[Strategy] | None = None,
    ) -> dict[str, list[float]]:
        """Shortcut for ``evaluate(...).scores``."""
        return self.evaluate(problem, strategies, seeds, opponents).scores
