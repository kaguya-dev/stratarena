"""Battle-royale arena: evaluate on fresh instances, eliminate the weakest, repeat."""

import math
import multiprocessing
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from stratarena.config import ArenaConfig
from stratarena.log import get_logger
from stratarena.problem import Problem
from stratarena.seeding import Stream, derive_seed, make_rng
from stratarena.strategy import ParamStrategy

log = get_logger(__name__)

Mapper = Callable[[Callable[[int], list[float]], Iterable[int]], Iterator[list[float]]]


@dataclass(frozen=True)
class Standing:
    """A strategy's mean score over every instance it has played, with a 95% bootstrap CI."""

    strategy_id: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class RoundRecord:
    index: int
    instance_seeds: tuple[int, ...]
    standings: tuple[Standing, ...]
    eliminated: tuple[str, ...]


@dataclass(frozen=True)
class ArenaResult:
    generation: int
    rounds: tuple[RoundRecord, ...]
    finalists: tuple[ParamStrategy, ...]
    standings: tuple[Standing, ...]
    scores: dict[str, tuple[float, ...]]


def evaluate_on_instance(
    problem: Problem[Any], strategies: Sequence[ParamStrategy], seed: int
) -> list[float]:
    """Score every strategy on the instance ``seed``. Runs inside worker processes."""
    instance = problem.generate_instance(seed)
    return [problem.evaluate(s, instance).score for s in strategies]


def bootstrap_standings(
    scores: dict[str, list[float]], rng: np.random.Generator, samples: int
) -> list[Standing]:
    """Rank strategies by mean score, best first; ties are broken by id.

    All strategies have played the same instances, so one resampling matrix is shared:
    the intervals are paired and directly comparable.
    """
    ids = sorted(scores)
    n = len(scores[ids[0]])
    idx = rng.integers(0, n, size=(samples, n))
    standings = []
    for sid in ids:
        arr = np.asarray(scores[sid], dtype=float)
        means = arr[idx].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        standings.append(Standing(sid, float(arr.mean()), float(low), float(high), n))
    standings.sort(key=lambda s: (-s.mean, s.strategy_id))
    return standings


class Arena:
    """Runs battle royales of parametric strategies on one problem.

    Each round draws ``instances_per_round`` new instances (seeded by generation and
    round), evaluates all survivors on the same instances, ranks them by mean score over
    all rounds so far and eliminates ``elimination_fraction`` of them, stopping once only
    ``finalists`` remain or ``max_rounds`` is reached.
    """

    def __init__(self, problem: Problem[Any], config: ArenaConfig, seed: int) -> None:
        self.problem = problem
        self.config = config
        self.seed = seed

    def run(self, candidates: Sequence[ParamStrategy], generation: int = 0) -> ArenaResult:
        ids = [c.id for c in candidates]
        if not candidates:
            raise ValueError("the arena needs at least one candidate")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be unique")

        cfg = self.config
        survivors = list(candidates)
        scores: dict[str, list[float]] = {c.id: [] for c in candidates}
        rounds: list[RoundRecord] = []
        standings: list[Standing] = []

        with self._mapper() as mapper:
            for r in range(cfg.max_rounds):
                seeds = tuple(
                    derive_seed(self.seed, Stream.TRAIN, generation, r, i)
                    for i in range(cfg.instances_per_round)
                )
                task = partial(evaluate_on_instance, self.problem, survivors)
                for per_instance in mapper(task, seeds):
                    for strategy, score in zip(survivors, per_instance, strict=True):
                        scores[strategy.id].append(score)

                rng = make_rng(self.seed, Stream.BOOTSTRAP, generation, r)
                alive = {s.id: scores[s.id] for s in survivors}
                standings = bootstrap_standings(alive, rng, cfg.bootstrap_samples)

                n_drop = 0
                if len(survivors) > cfg.finalists:
                    n_drop = max(1, math.floor(len(survivors) * cfg.elimination_fraction))
                    n_drop = min(n_drop, len(survivors) - cfg.finalists)
                eliminated = tuple(s.strategy_id for s in standings[len(standings) - n_drop :])
                survivors = [s for s in survivors if s.id not in eliminated]
                standings = standings[: len(standings) - n_drop]

                rounds.append(RoundRecord(r, seeds, tuple(standings), eliminated))
                log.info(
                    "arena.round",
                    generation=generation,
                    round=r,
                    survivors=len(survivors),
                    eliminated=len(eliminated),
                    best=standings[0].strategy_id,
                    best_mean=round(standings[0].mean, 6),
                )
                if len(survivors) <= cfg.finalists:
                    break

        by_id = {s.id: s for s in survivors}
        return ArenaResult(
            generation=generation,
            rounds=tuple(rounds),
            finalists=tuple(by_id[s.strategy_id] for s in standings),
            standings=tuple(standings),
            scores={sid: tuple(v) for sid, v in scores.items()},
        )

    @contextmanager
    def _mapper(self) -> Iterator[Mapper]:
        if self.config.workers == 1:
            yield map
            return
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.config.workers, mp_context=ctx) as pool:
            yield pool.map
