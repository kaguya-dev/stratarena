import pytest

from stratarena.examples.binpacking import SPACE, BinPacking
from stratarena.strategy import ParamStrategy


@pytest.fixture
def problem() -> BinPacking:
    return BinPacking(n_items=60, capacity=100)


def make(id: str, sort_fraction: float, fit_weight: float, first_weight: float) -> ParamStrategy:
    return ParamStrategy.from_mapping(
        id,
        SPACE,
        {
            "sort_fraction": sort_fraction,
            "fit_weight": fit_weight,
            "first_weight": first_weight,
            "exact_bonus": 0.0,
            "waste_penalty": 0.0,
            "fillable_bonus": 0.0,
            "open_penalty": 5.0,
        },
    )


@pytest.fixture
def ffd() -> ParamStrategy:
    """First fit decreasing."""
    return make("ffd", 1.0, 0.0, 1.0)


@pytest.fixture
def bfd() -> ParamStrategy:
    """Best fit decreasing."""
    return make("bfd", 1.0, 1.0, 0.0)


@pytest.fixture
def worst_online() -> ParamStrategy:
    """Worst fit in arrival order: a deliberately weak heuristic."""
    return make("worst", 0.0, -1.0, 0.0)
