import numpy as np
import pytest

from stratarena.arena import Arena, bootstrap_standings
from stratarena.config import ArenaConfig
from stratarena.examples.binpacking import SPACE, BinPacking
from stratarena.seeding import Stream, make_rng
from stratarena.strategy import ParamStrategy

FAST = ArenaConfig(instances_per_round=4, finalists=3, max_rounds=10, bootstrap_samples=200)


def population(n: int, seed: int = 0) -> list[ParamStrategy]:
    rng = make_rng(seed, Stream.INIT, 0)
    return [ParamStrategy.random(f"s{i:02d}", SPACE, rng) for i in range(n)]


def test_eliminates_down_to_finalists(problem: BinPacking) -> None:
    result = Arena(problem, FAST, seed=1).run(population(12))
    assert len(result.finalists) == 3
    assert len(result.standings) == 3
    eliminated = [sid for r in result.rounds for sid in r.eliminated]
    assert len(eliminated) == len(set(eliminated)) == 9
    assert not {f.id for f in result.finalists} & set(eliminated)


def test_standings_are_sorted_and_ci_contains_mean(problem: BinPacking) -> None:
    result = Arena(problem, FAST, seed=1).run(population(8))
    for record in result.rounds:
        means = [s.mean for s in record.standings]
        assert means == sorted(means, reverse=True)
        for s in record.standings:
            assert s.ci_low <= s.mean <= s.ci_high


def test_all_survivors_play_the_same_instances(problem: BinPacking) -> None:
    result = Arena(problem, FAST, seed=1).run(population(8))
    lengths = {len(result.scores[f.id]) for f in result.finalists}
    assert lengths == {FAST.instances_per_round * len(result.rounds)}


def test_same_seed_same_result(problem: BinPacking) -> None:
    a = Arena(problem, FAST, seed=3).run(population(10))
    b = Arena(problem, FAST, seed=3).run(population(10))
    assert a == b


def test_different_seed_different_instances(problem: BinPacking) -> None:
    a = Arena(problem, FAST, seed=3).run(population(10))
    b = Arena(problem, FAST, seed=4).run(population(10))
    assert a.rounds[0].instance_seeds != b.rounds[0].instance_seeds


def test_parallel_matches_sequential(problem: BinPacking) -> None:
    seq = Arena(problem, FAST, seed=5).run(population(8))
    par = Arena(problem, FAST.model_copy(update={"workers": 2}), seed=5).run(population(8))
    assert seq == par


def test_weak_strategy_is_eliminated_first(
    problem: BinPacking, ffd: ParamStrategy, bfd: ParamStrategy, worst_online: ParamStrategy
) -> None:
    cfg = ArenaConfig(instances_per_round=6, finalists=2, bootstrap_samples=200)
    result = Arena(problem, cfg, seed=0).run([worst_online, ffd, bfd])
    assert result.rounds[0].eliminated == ("worst",)
    assert {f.id for f in result.finalists} == {"ffd", "bfd"}


def test_no_elimination_when_already_at_finalists(problem: BinPacking) -> None:
    result = Arena(problem, FAST, seed=0).run(population(2))
    assert len(result.rounds) == 1
    assert result.rounds[0].eliminated == ()


def test_rejects_bad_candidates(problem: BinPacking) -> None:
    arena = Arena(problem, FAST, seed=0)
    with pytest.raises(ValueError, match="at least one"):
        arena.run([])
    dup = population(2)
    with pytest.raises(ValueError, match="unique"):
        arena.run([dup[0], dup[0]])


def test_bootstrap_ties_break_by_id() -> None:
    rng = np.random.default_rng(0)
    standings = bootstrap_standings({"b": [1.0, 1.0], "a": [1.0, 1.0]}, rng, 100)
    assert [s.strategy_id for s in standings] == ["a", "b"]
