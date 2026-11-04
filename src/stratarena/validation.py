"""Held-out validation: score the finalists on instances the search never saw."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from stratarena.arena import Standing, bootstrap_standings
from stratarena.evaluation import Evaluator
from stratarena.problem import AnyProblem, GameProblem
from stratarena.seeding import Stream, derive_seed, make_rng
from stratarena.stats import mean_ci, resample_indices
from stratarena.strategy import ParamStrategy

# Bootstrap path for validation, far from any (generation, round) path used by the arena.
_HOLDOUT_BOOTSTRAP_KEY = 2**31


@dataclass(frozen=True)
class HoldoutEntry:
    """One strategy on the held-out set.

    ``diff_*`` is the paired difference to the champion (strategy minus champion), with a
    95% bootstrap interval: an interval entirely above zero means the strategy beats the
    champion on unseen instances.
    """

    strategy_id: str
    kind: str
    train_mean: float | None
    holdout: Standing
    diff_mean: float
    diff_low: float
    diff_high: float
    rating: float | None = None
    """Elo-scale rating from the held-out round robin (games only)."""


@dataclass(frozen=True)
class ValidationResult:
    instance_seeds: tuple[int, ...]
    champion_id: str
    entries: tuple[HoldoutEntry, ...]

    @property
    def champion(self) -> HoldoutEntry:
        return next(e for e in self.entries if e.strategy_id == self.champion_id)

    @property
    def champion_rank(self) -> int:
        """1-based rank of the champion on the held-out set, among finalists only."""
        finalists = [e.strategy_id for e in self.entries if e.kind == "finalist"]
        return finalists.index(self.champion_id) + 1

    @property
    def generalization_gap(self) -> float:
        """Champion's training mean minus held-out mean; large values hint at overfitting."""
        champ = self.champion
        assert champ.train_mean is not None
        return champ.train_mean - champ.holdout.mean

    @property
    def overfit_suspected(self) -> bool:
        """True when the champion's training mean lies above its held-out 95% interval."""
        champ = self.champion
        assert champ.train_mean is not None
        return champ.train_mean > champ.holdout.ci_high


def validate(
    problem: AnyProblem,
    finalists: Sequence[ParamStrategy],
    train_means: Mapping[str, float],
    baselines: Sequence[ParamStrategy],
    seed: int,
    n_instances: int,
    bootstrap_samples: int,
    evaluator: Evaluator,
) -> ValidationResult:
    """Evaluate finalists (the first is the champion) and baselines on held-out instances.

    For a game, everyone plays a round robin against everyone else (finalists and
    baselines), and the score is the mean points per game against that field.
    """
    if not finalists:
        raise ValueError("need at least one finalist")
    seeds = tuple(derive_seed(seed, Stream.HOLDOUT, i) for i in range(n_instances))
    finalist_ids = {f.id for f in finalists}
    extra = [b for b in baselines if b.id not in finalist_ids]
    evaluation = evaluator.evaluate(problem, [*finalists, *extra], seeds)
    scores = evaluation.scores
    ratings = evaluation.pairs.ratings(scores) if isinstance(problem, GameProblem) else {}

    rng = make_rng(seed, Stream.BOOTSTRAP, _HOLDOUT_BOOTSTRAP_KEY)
    standings = bootstrap_standings(scores, rng, bootstrap_samples)
    idx = resample_indices(rng, n_instances, bootstrap_samples)
    champion_id = finalists[0].id
    champ = np.asarray(scores[champion_id])

    entries = []
    for s in standings:
        diff = (np.asarray(scores[s.strategy_id]) - champ).tolist()
        d_mean, d_low, d_high = mean_ci(diff, idx)
        is_finalist = s.strategy_id in finalist_ids
        entries.append(
            HoldoutEntry(
                strategy_id=s.strategy_id,
                kind="finalist" if is_finalist else "baseline",
                train_mean=train_means.get(s.strategy_id) if is_finalist else None,
                holdout=s,
                diff_mean=d_mean,
                diff_low=d_low,
                diff_high=d_high,
                rating=ratings.get(s.strategy_id),
            )
        )
    return ValidationResult(seeds, champion_id, tuple(entries))
