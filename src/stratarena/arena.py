"""Battle-royale arena: evaluate on fresh instances, eliminate the weakest, repeat."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stratarena.config import ArenaConfig
from stratarena.evaluation import Evaluator
from stratarena.log import get_logger
from stratarena.problem import AnyProblem, GameProblem
from stratarena.rating import PairStats
from stratarena.seeding import Stream, derive_seed, make_rng
from stratarena.stats import mean_ci, resample_indices
from stratarena.strategy import Strategy

log = get_logger(__name__)


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
    finalists: tuple[Strategy, ...]
    standings: tuple[Standing, ...]
    scores: dict[str, tuple[float, ...]]
    ratings: dict[str, float]
    """Elo-scale ratings of the finalists from their games against each other (games only)."""


def bootstrap_standings(
    scores: dict[str, list[float]], rng: np.random.Generator, samples: int
) -> list[Standing]:
    """Rank strategies by mean score, best first; ties are broken by id.

    All strategies have played the same instances, so one resampling matrix is shared:
    the intervals are paired and directly comparable.
    """
    ids = sorted(scores)
    n = len(scores[ids[0]])
    idx = resample_indices(rng, n, samples)
    standings = [Standing(sid, *mean_ci(scores[sid], idx), n) for sid in ids]
    standings.sort(key=lambda s: (-s.mean, s.strategy_id))
    return standings


class Arena:
    """Runs battle royales of parametric strategies on one problem.

    Each round draws ``instances_per_round`` new instances (seeded by generation and
    round), evaluates all survivors on the same instances, ranks them by mean score over
    all rounds so far and eliminates ``elimination_fraction`` of them, stopping once only
    ``finalists`` remain or ``max_rounds`` is reached.

    For a game the score on an instance is the mean points per game against every other
    survivor (round robin, both sides), so the ranking is by head-to-head wins; Elo-scale
    ratings of the finalists are reported alongside.
    """

    def __init__(
        self,
        problem: AnyProblem,
        config: ArenaConfig,
        seed: int,
        sparring: Sequence[Strategy] = (),
    ) -> None:
        """``sparring``: fixed opponents (games only) that join every round robin.

        They are played against but never ranked or eliminated, so the population cannot
        drift into only beating itself.
        """
        if sparring and not isinstance(problem, GameProblem):
            raise ValueError("sparring partners only make sense for game problems")
        self.problem = problem
        self.config = config
        self.seed = seed
        self.sparring = list(sparring)

    def run(
        self,
        candidates: Sequence[Strategy],
        generation: int = 0,
        evaluator: Evaluator | None = None,
    ) -> ArenaResult:
        ids = [c.id for c in candidates]
        if not candidates:
            raise ValueError("the arena needs at least one candidate")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be unique")
        if set(ids) & {s.id for s in self.sparring}:
            raise ValueError("a candidate has the same id as a sparring partner")
        if evaluator is None:
            with Evaluator(self.config.workers) as own:
                return self._run(candidates, generation, own)
        return self._run(candidates, generation, evaluator)

    def _run(
        self, candidates: Sequence[Strategy], generation: int, evaluator: Evaluator
    ) -> ArenaResult:
        cfg = self.config
        survivors = list(candidates)
        scores: dict[str, list[float]] = {c.id: [] for c in candidates}
        rounds: list[RoundRecord] = []
        standings: list[Standing] = []
        pairs = PairStats()

        for r in range(cfg.max_rounds):
            seeds = tuple(
                derive_seed(self.seed, Stream.TRAIN, generation, r, i)
                for i in range(cfg.instances_per_round)
            )
            evaluation = evaluator.evaluate(self.problem, [*survivors, *self.sparring], seeds)
            pairs.merge(evaluation.pairs)
            for s in survivors:
                scores[s.id].extend(evaluation.scores[s.id])

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
            ratings=pairs.ratings(by_id) if isinstance(self.problem, GameProblem) else {},
        )
