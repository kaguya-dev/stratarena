import numpy as np
import pytest

from stratarena.strategy import ParamSpec, ParamStrategy, clip_to_space

SPACE = (ParamSpec("a", 0.0, 1.0), ParamSpec("b", -5.0, 5.0))


def test_random_strategy_is_inside_bounds() -> None:
    rng = np.random.default_rng(0)
    for i in range(100):
        s = ParamStrategy.random(f"s{i}", SPACE, rng)
        assert 0.0 <= s["a"] <= 1.0
        assert -5.0 <= s["b"] <= 5.0


def test_out_of_bounds_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        ParamStrategy("x", (2.0, 0.0), SPACE)


def test_wrong_number_of_values_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        ParamStrategy("x", (0.5,), SPACE)


def test_invalid_spec_is_rejected() -> None:
    with pytest.raises(ValueError):
        ParamSpec("bad", 1.0, 1.0)


def test_from_mapping_round_trips() -> None:
    s = ParamStrategy.from_mapping("x", SPACE, {"a": 0.25, "b": -1.0})
    assert s.as_dict() == {"a": 0.25, "b": -1.0}
    with pytest.raises(KeyError):
        s["missing"]


def test_from_mapping_requires_every_parameter() -> None:
    with pytest.raises(ValueError, match="missing"):
        ParamStrategy.from_mapping("x", SPACE, {"a": 0.5})


def test_clip_to_space() -> None:
    assert clip_to_space(SPACE, [1.5, -9.0]) == (1.0, -5.0)
    with pytest.raises(ValueError):
        clip_to_space(SPACE, [0.5])
