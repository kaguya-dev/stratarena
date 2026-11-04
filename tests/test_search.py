import pytest

from stratarena.config import ArenaConfig, HybridConfig, RunConfig
from stratarena.evaluation import Evaluator
from stratarena.examples.binpacking import BinPacking
from stratarena.search import RunResult, run_search

SMALL = RunConfig(
    seed=11,
    problem="binpacking",
    population=8,
    generations=3,
    holdout_instances=12,
    arena=ArenaConfig(instances_per_round=3, finalists=2, bootstrap_samples=100),
    hybrid=HybridConfig(immigrants=1),
)


@pytest.fixture(scope="module")
def result() -> RunResult:
    return run_search(BinPacking(n_items=30), SMALL)


def test_runs_every_generation(result: RunResult) -> None:
    assert len(result.generations) == len(result.populations) == 3
    assert all(len(p) == 8 for p in result.populations)


def test_elitism_keeps_finalists(result: RunResult) -> None:
    for prev, nxt in zip(result.generations, result.populations[1:], strict=False):
        assert {f.id for f in prev.finalists} <= {s.id for s in nxt}


def test_offspring_and_immigrants_are_added(result: RunResult) -> None:
    ids = [s.id for s in result.populations[1]]
    assert sum(i.startswith("g1-c") for i in ids) == 5
    assert sum(i.startswith("g1-r") for i in ids) == 1


def test_holdout_never_reuses_training_instances(result: RunResult) -> None:
    train = {s for g in result.generations for r in g.rounds for s in r.instance_seeds}
    assert train.isdisjoint(result.validation.instance_seeds)


def test_validation_compares_against_baselines(result: RunResult) -> None:
    v = result.validation
    assert v.champion_id == result.champion.id
    assert v.champion.diff_mean == 0.0
    kinds = {e.strategy_id: e.kind for e in v.entries}
    assert kinds["first-fit-decreasing"] == "baseline"
    assert kinds[result.champion.id] == "finalist"
    assert 1 <= v.champion_rank <= 2
    assert isinstance(v.overfit_suspected, bool)


def test_search_is_deterministic(result: RunResult) -> None:
    assert run_search(BinPacking(n_items=30), SMALL) == result


def test_baselines_can_compete() -> None:
    cfg = SMALL.model_copy(update={"baselines_compete": True, "generations": 1})
    res = run_search(BinPacking(n_items=30), cfg)
    assert "best-fit-decreasing" in {s.id for s in res.populations[0]}
    assert len(res.populations[0]) == 8


def test_config_needs_room_for_offspring() -> None:
    with pytest.raises(ValueError, match="population must exceed"):
        RunConfig(problem="binpacking", population=4, arena=ArenaConfig(finalists=3))


def test_evaluator_needs_context_for_workers() -> None:
    with pytest.raises(RuntimeError):
        Evaluator(workers=2).scores(BinPacking(n_items=5), [], [1])
    with pytest.raises(ValueError):
        Evaluator(workers=0)


def test_tracking_is_fixed_and_separate(result: RunResult) -> None:
    assert len(result.tracking) == len(result.generations)
    for g, t in zip(result.generations, result.tracking, strict=True):
        assert set(t) == {f.id for f in g.finalists}
    # A finalist kept by elitism scores the same on the fixed tracking set every generation.
    kept = set(result.tracking[0]) & set(result.tracking[1])
    for sid in kept:
        assert result.tracking[0][sid] == result.tracking[1][sid]
    assert set(result.baseline_tracking) == {b.id for b in result.baselines}
