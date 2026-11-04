import pytest

from stratarena.evaluation import Evaluator
from stratarena.examples.binpacking import CODE_EXAMPLE as PACK_EXAMPLE
from stratarena.examples.binpacking import BinPacking
from stratarena.examples.gridbots import CODE_EXAMPLE as BOT_EXAMPLE
from stratarena.examples.gridbots import GridBots
from stratarena.strategy import CodeStrategy


def code(id: str, source: str, entrypoint: str) -> CodeStrategy:
    return CodeStrategy(id, source, entrypoint)


def test_reference_code_matches_parametric_ffd() -> None:
    problem = BinPacking(n_items=60)
    ffd = next(b for b in problem.baselines() if b.id == "first-fit-decreasing")
    ffd_code = code("ffd-code", PACK_EXAMPLE, "pack")
    for seed in range(4):
        inst = problem.generate_instance(seed)
        assert problem.evaluate(ffd_code, inst).score == problem.evaluate(ffd, inst).score


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("def pack(items, capacity):\n    return [0] * len(items)\n", "capacity"),
        ("def pack(items, capacity):\n    return [0]\n", "one bin number"),
        ("def pack(items, capacity):\n    return [-1] * len(items)\n", "invalid bin"),
        ("def pack(items, capacity):\n    return 1 / 0\n", "ZeroDivisionError"),
    ],
)
def test_invalid_packings_score_zero(source: str, reason: str) -> None:
    problem = BinPacking(n_items=20)
    result = problem.evaluate(code("bad", source, "pack"), problem.generate_instance(1))
    assert result.score == 0.0
    assert result.error is not None and reason in result.error


def test_code_bot_plays_and_rule_breakers_forfeit() -> None:
    game = GridBots(size=9, coin_pairs=6, turns=40)
    inst = game.generate_instance(2)
    bfs = code("bfs", BOT_EXAMPLE, "choose_move")
    near = game.baselines()[0]
    assert game.play(bfs, near, inst) in (0.0, 0.5, 1.0)
    bad_move = code("bad", "def choose_move(state):\n    return 7\n", "choose_move")
    crash = code("crash", "def choose_move(state):\n    raise ValueError\n", "choose_move")
    broken = code("broken", "def choose_move(:\n", "choose_move")
    for bot in (bad_move, crash, broken):
        assert game.play(bot, near, inst) == 0.0
        assert game.play(near, bot, inst) == 1.0


def test_code_bot_state_is_complete() -> None:
    probe = (
        "def choose_move(state):\n"
        "    keys = {'grid', 'me', 'opponent', 'coins', 'turn', 'turns_left',"
        " 'my_points', 'opponent_points'}\n"
        "    assert set(state) == keys, set(state)\n"
        "    assert state['grid'][state['me'][0]][state['me'][1]] == '.'\n"
        "    return 4\n"
    )
    game = GridBots(size=7, coin_pairs=3, turns=5)
    near = game.baselines()[0]
    result = game.play(code("probe", probe, "choose_move"), near, game.generate_instance(0))
    assert result in (0.0, 0.5)  # staying put never wins, but it must not forfeit early


def test_code_strategies_work_on_worker_processes() -> None:
    problem = BinPacking(n_items=30)
    s = code("ffd-code", PACK_EXAMPLE, "pack")
    with Evaluator(workers=2) as ev:
        par = ev.scores(problem, [s], [1, 2, 3])
    assert par == Evaluator().scores(problem, [s], [1, 2, 3])


def test_digest_tracks_source() -> None:
    a = code("a", PACK_EXAMPLE, "pack")
    assert a.digest == code("b", PACK_EXAMPLE, "pack").digest
    assert a.digest != code("c", PACK_EXAMPLE + "\n", "pack").digest
