from pathlib import Path

import pytest

from stratarena.arena import Arena
from stratarena.config import ArenaConfig, HybridConfig, RunConfig
from stratarena.evaluation import Evaluator
from stratarena.examples import make_problem
from stratarena.examples.gridbots import SPACE, GridBots, GridInstance, _geometry
from stratarena.report import write_run
from stratarena.search import run_search
from stratarena.strategy import ParamStrategy


@pytest.fixture(scope="module")
def game() -> GridBots:
    return GridBots(size=9, coin_pairs=8, turns=50)


def bot(id: str, *values: float) -> ParamStrategy:
    return ParamStrategy(id, values, SPACE)


GREEDY = bot("greedy", 1.0, 0.5, 0.0, 0.0, 0.0)
LAZY = bot("lazy", -1.0, -1.0, 0.0, -1.0, 0.0)


def test_maps_are_symmetric_connected_and_deterministic(game: GridBots) -> None:
    for seed in range(15):
        inst = game.generate_instance(seed)
        n = inst.height * inst.width
        assert inst == game.generate_instance(seed)
        assert all(inst.walls[c] == inst.walls[n - 1 - c] for c in range(n))
        coins = dict(inst.coins)
        assert all(coins[n - 1 - c] == v for c, v in coins.items())
        a, b = inst.starts
        dist, _ = _geometry(inst)
        free = [c for c in range(n) if not inst.walls[c]]
        assert all(dist[a, c] < n for c in free)
        assert dist[a, b] < n


def test_moves_never_enter_walls(game: GridBots) -> None:
    inst = game.generate_instance(3)
    _, moves = _geometry(inst)
    for cell in range(inst.height * inst.width):
        if not inst.walls[cell]:
            assert not any(inst.walls[int(t)] for t in moves[cell])


def test_play_is_deterministic_and_valued(game: GridBots) -> None:
    inst = game.generate_instance(4)
    result = game.play(GREEDY, LAZY, inst)
    assert result in (0.0, 0.5, 1.0)
    assert result == game.play(GREEDY, LAZY, inst)


def test_greedy_beats_coin_avoider(game: GridBots) -> None:
    insts = [game.generate_instance(s) for s in range(10)]
    pts = sum(game.play(GREEDY, LAZY, i) + 1 - game.play(LAZY, GREEDY, i) for i in insts)
    assert pts / 20 > 0.9


def test_empty_board_is_a_draw(game: GridBots) -> None:
    inst = game.generate_instance(0)
    empty = GridInstance(inst.height, inst.width, inst.walls, (), inst.turns)
    assert game.play(GREEDY, LAZY, empty) == 0.5


def test_round_robin_is_zero_sum(game: GridBots) -> None:
    bots = [GREEDY, LAZY, *game.baselines()]
    ev = Evaluator().evaluate(game, bots, [1, 2])
    for i in range(2):
        assert sum(ev.scores[b.id][i] for b in bots) == pytest.approx(len(bots) / 2)
    assert ev.pairs.games[("greedy", "lazy")] == 4


def test_scores_against_fixed_opponents(game: GridBots) -> None:
    scores = Evaluator().scores(game, [GREEDY, LAZY], [1, 2], opponents=game.baselines())
    assert sum(scores["greedy"]) > sum(scores["lazy"])
    alone = Evaluator().scores(game, [GREEDY], [1], opponents=[GREEDY])
    assert alone["greedy"] == [0.5]


def test_arena_ranks_by_head_to_head_and_rates_finalists(game: GridBots) -> None:
    cfg = ArenaConfig(instances_per_round=3, finalists=2, bootstrap_samples=100)
    result = Arena(game, cfg, seed=0).run([LAZY, GREEDY, *game.baselines()])
    assert "lazy" in result.rounds[0].eliminated
    assert set(result.ratings) == {f.id for f in result.finalists}


def test_game_search_end_to_end(tmp_path: Path) -> None:
    cfg = RunConfig(
        seed=1,
        problem="gridbots",
        problem_options={"size": 7, "coin_pairs": 4, "turns": 25},
        population=6,
        generations=2,
        holdout_instances=4,
        tracking_instances=2,
        arena=ArenaConfig(instances_per_round=2, finalists=2, bootstrap_samples=100),
        hybrid=HybridConfig(immigrants=1),
    )
    problem = make_problem(cfg.problem, cfg.problem_options)
    result = run_search(problem, cfg)
    assert result.mode == "game"
    assert set(result.baseline_tracking) == {b.id for b in GridBots().baselines()}
    entries = {e.strategy_id: e for e in result.validation.entries}
    assert all(e.rating is not None for e in entries.values())
    assert run_search(problem, cfg) == result

    md = (write_run(result, tmp_path) / "report.md").read_text()
    assert "Elo |" in md
    assert "not an overfitting measure" in md


def test_invalid_options() -> None:
    with pytest.raises(ValueError):
        GridBots(size=2)


def test_sparring_partners_play_but_are_never_ranked(game: GridBots) -> None:
    cfg = ArenaConfig(instances_per_round=2, finalists=2, bootstrap_samples=100)
    sparring = game.baselines()
    result = Arena(game, cfg, seed=0, sparring=sparring).run([GREEDY, LAZY, bot("mid", *[0.2] * 5)])
    ranked = {s.strategy_id for r in result.rounds for s in r.standings}
    assert ranked.isdisjoint({s.id for s in sparring})
    assert set(result.scores) == {"greedy", "lazy", "mid"}
    plain = Arena(game, cfg, seed=0).run([GREEDY, LAZY, bot("mid", *[0.2] * 5)])
    assert result.scores["greedy"] != plain.scores["greedy"]


def test_sparring_rules() -> None:
    from stratarena.examples.binpacking import BinPacking

    with pytest.raises(ValueError, match="game"):
        Arena(BinPacking(), ArenaConfig(), 0, sparring=BinPacking().baselines())
    game = GridBots(size=7)
    with pytest.raises(ValueError, match="sparring partner"):
        Arena(game, ArenaConfig(), 0, sparring=game.baselines()).run(game.baselines())
    with pytest.raises(ValueError, match="compete or spar"):
        RunConfig(problem="gridbots", sparring=True, baselines_compete=True)
