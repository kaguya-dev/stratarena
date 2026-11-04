from pathlib import Path

import pytest
from pydantic import ValidationError

from stratarena.config import ArenaConfig, RunConfig, load_config

ROOT = Path(__file__).parent.parent


def test_shipped_config_is_valid() -> None:
    cfg = load_config(ROOT / "configs" / "binpacking.yaml")
    assert cfg.problem == "binpacking"
    assert cfg.arena.finalists == 4


def test_defaults() -> None:
    cfg = RunConfig(problem="binpacking")
    assert cfg.seed == 0
    assert cfg.arena == ArenaConfig()


@pytest.mark.parametrize(
    "arena",
    [{"elimination_fraction": 1.0}, {"finalists": 0}, {"workers": 0}, {"typo": 1}],
)
def test_invalid_arena_options_are_rejected(arena: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"problem": "binpacking", "arena": arena})


def test_load_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("problem: binpacking\nseed: 9\narena:\n  finalists: 2\n")
    cfg = load_config(path)
    assert cfg.seed == 9
    assert cfg.arena.finalists == 2
