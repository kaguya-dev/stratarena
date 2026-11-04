import pytest

from stratarena.config import HybridConfig
from stratarena.examples.binpacking import SPACE
from stratarena.hybridizer import ParamHybridizer
from stratarena.strategy import ParamStrategy

A = ParamStrategy("a", (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0), SPACE)
B = ParamStrategy("b", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0), SPACE)


def test_children_are_valid_and_record_parents() -> None:
    children = ParamHybridizer(HybridConfig(mutation_rate=1.0), seed=0).offspring([A, B], 20, 3)
    assert [c.id for c in children] == [f"g4-c{i:02d}" for i in range(20)]
    for c in children:
        assert set(c.parents) == {"a", "b"}
        for v, spec in zip(c.values, SPACE, strict=True):
            assert spec.low <= v <= spec.high


def test_offspring_are_reproducible() -> None:
    h = ParamHybridizer(HybridConfig(), seed=1)
    assert h.offspring([A, B], 5, 0) == h.offspring([A, B], 5, 0)
    assert h.offspring([A, B], 5, 0) != h.offspring([A, B], 5, 1)


def test_without_mutation_genes_stay_in_blend_interval() -> None:
    h = ParamHybridizer(HybridConfig(crossover_alpha=0.0, mutation_rate=0.0), seed=2)
    for c in h.offspring([A, B], 10, 0):
        for v, x, y in zip(c.values, A.values, B.values, strict=True):
            assert min(x, y) <= v <= max(x, y)


def test_single_parent_is_mutated_copy() -> None:
    h = ParamHybridizer(HybridConfig(mutation_rate=0.0), seed=0)
    (child,) = h.offspring([B], 1, 0)
    assert child.values == B.values
    assert child.parents == ("b",)


def test_needs_parents() -> None:
    with pytest.raises(ValueError):
        ParamHybridizer(HybridConfig(), seed=0).offspring([], 1, 0)
