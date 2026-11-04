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
