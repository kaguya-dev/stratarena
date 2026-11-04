from collections import Counter

import pytest

from stratarena.examples import make_problem
from stratarena.examples.binpacking import BinPacking
from stratarena.strategy import ParamStrategy


def test_instances_are_deterministic(problem: BinPacking) -> None:
    assert problem.generate_instance(5) == problem.generate_instance(5)
    assert problem.generate_instance(5) != problem.generate_instance(6)


def test_items_fit_in_a_bin(problem: BinPacking) -> None:
    for seed in range(20):
        inst = problem.generate_instance(seed)
        assert len(inst.items) == 60
        assert all(1 <= x <= inst.capacity for x in inst.items)


@pytest.mark.parametrize("name", ["ffd", "bfd", "worst_online"])
def test_packing_is_valid(problem: BinPacking, name: str, request: pytest.FixtureRequest) -> None:
    strategy: ParamStrategy = request.getfixturevalue(name)
    for seed in range(10):
        inst = problem.generate_instance(seed)
        bins = problem.pack(strategy, inst)
        assert all(sum(b) <= inst.capacity for b in bins)
        assert Counter(x for b in bins for x in b) == Counter(inst.items)
        result = problem.evaluate(strategy, inst)
        assert 0.0 < result.score <= 1.0
        assert result.metrics["bins"] >= result.metrics["lower_bound"]


def test_decreasing_heuristics_beat_worst_fit_online(
    problem: BinPacking, ffd: ParamStrategy, worst_online: ParamStrategy
) -> None:
    instances = [problem.generate_instance(s) for s in range(30)]
    good = sum(problem.evaluate(ffd, i).score for i in instances)
    bad = sum(problem.evaluate(worst_online, i).score for i in instances)
    assert good > bad


def test_registry() -> None:
    p = make_problem("binpacking", {"n_items": 10})
    assert isinstance(p, BinPacking)
    assert p.n_items == 10
    with pytest.raises(ValueError, match="unknown problem"):
        make_problem("nope")


def test_invalid_options() -> None:
    with pytest.raises(ValueError):
        BinPacking(n_items=0)


def test_baselines_ignore_lookahead_features(problem: BinPacking) -> None:
    names = {b.id for b in problem.baselines()}
    assert names == {"first-fit", "best-fit", "first-fit-decreasing", "best-fit-decreasing"}
    for b in problem.baselines():
        assert b["exact_bonus"] == b["waste_penalty"] == b["fillable_bonus"] == 0.0
        assert b["open_penalty"] == 5.0


def test_triplets_have_a_known_perfect_packing(problem: BinPacking) -> None:
    triplets = [
        i for i in (problem.generate_instance(s) for s in range(40)) if i.optimum is not None
    ]
    assert triplets
    for inst in triplets:
        assert inst.optimum is not None
        assert len(inst.items) == 3 * inst.optimum
        assert sum(inst.items) == inst.optimum * inst.capacity
        assert inst.reference == inst.optimum == inst.lower_bound


def test_opening_bins_on_purpose_beats_ffd_on_triplets(
    problem: BinPacking, ffd: ParamStrategy
) -> None:
    triplets = [
        i for i in (problem.generate_instance(s) for s in range(80)) if i.optimum is not None
    ]
    smart = ParamStrategy("smart", (0.07, 0.37, 0.8, 0.64, 0.34, 0.56, 0.11), problem.param_space())
    ffd_score = sum(problem.evaluate(ffd, i).score for i in triplets)
    smart_score = sum(problem.evaluate(smart, i).score for i in triplets)
    assert smart_score > ffd_score


def test_lookahead_strategy_packs_validly(problem: BinPacking) -> None:
    s = ParamStrategy("look", (1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.5), problem.param_space())
    for seed in range(10):
        inst = problem.generate_instance(seed)
        bins = problem.pack(s, inst)
        assert all(sum(b) <= inst.capacity for b in bins)
        assert Counter(x for b in bins for x in b) == Counter(inst.items)


def test_profiles_can_be_restricted() -> None:
    p = BinPacking(n_items=30, profiles=["triplets"])
    assert all(p.generate_instance(s).optimum == 10 for s in range(10))
    with pytest.raises(ValueError, match="profiles"):
        BinPacking(profiles=["nope"])
    with pytest.raises(ValueError, match="profiles"):
        BinPacking(profiles=[])
