"""Grid Bots: a small two-player coin race, designed for stratarena.

Two bots start in opposite corners of a grid with walls and coins (worth 1 or 3). Each
turn both choose a move (north, south, west, east or stay) at the same time. A bot that
ends its move on a coin collects it; if both try to enter the same cell, neither moves.
After ``turns`` turns, or once every coin is taken, the bot with more points wins.

Maps are point-symmetric (rotating by 180 degrees swaps the two bots' views), and the
arena plays each pair on both sides, so neither starting corner is an advantage.

A strategy scores each of its five possible moves with a weighted sum of features of the
cell it would move to, then plays the best one (ties go to the first move in N, S, W, E,
stay order). All weights are in ``[-1, 1]``:

- ``near``: closeness of the nearest coin.
- ``value``: best value-over-distance among the coins.
- ``race``: share of coin value the bot is strictly closer to than the opponent.
- ``density``: value of nearby coins, discounted with distance.
- ``spacing``: distance to the opponent (positive keeps away, negative chases).
"""

import contextlib
import functools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from stratarena.problem import CodeSpec
from stratarena.sandbox import SandboxedFunction, SandboxError
from stratarena.strategy import CodeStrategy, ParamSpace, ParamSpec, ParamStrategy, Strategy

SPACE: ParamSpace = (
    ParamSpec("near", -1.0, 1.0),
    ParamSpec("value", -1.0, 1.0),
    ParamSpec("race", -1.0, 1.0),
    ParamSpec("density", -1.0, 1.0),
    ParamSpec("spacing", -1.0, 1.0),
)

# Move order also breaks ties: N, S, W, E, stay.
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

CODE_DESCRIPTION = """\
Write a function `choose_move(state: dict) -> int`, called once per turn.

`state` has:
- `grid`: list of strings, one per row; `#` is a wall, `.` is free.
- `me`, `opponent`: `[row, col]` positions.
- `coins`: list of `[row, col, value]` still on the board (value 1 or 3).
- `turn`, `turns_left`, `my_points`, `opponent_points`: integers.

Return 0 = north (row - 1), 1 = south (row + 1), 2 = west (col - 1), 3 = east (col + 1)
or 4 = stay. Moving into a wall or off the grid means staying. Both bots move at the same
time; if both try to enter the same cell, neither moves. Ending a move on a coin collects
it. Most points after the last turn wins. Module-level variables persist during one game.
Standard library only, at most 2 seconds per call; an exception, timeout or invalid move
forfeits the game.
"""

PARAM_GUIDE = """\
A parametric bot scores each of its five moves (N, S, W, E, stay; ties go to the first)
by the cell it would move to, with shortest-path distances `d` around walls and
`scale = rows + cols`:

    score = near    * (-min over coins of d(cell, coin) / scale)
          + value   * (max over coins of value / (1 + d(cell, coin)) / 3)
          + race    * (value of coins strictly closer to me than to the opponent
                       - value of coins strictly closer to the opponent) / total value
          + density * (sum over coins of value * exp(-d(cell, coin) / 3)) / total value
          + spacing * d(cell, opponent) / scale

Positive `spacing` keeps away from the opponent, negative chases it.
"""

CODE_EXAMPLE = """\
from collections import deque

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def choose_move(state):
    \"\"\"Walk along a shortest path to the nearest coin.\"\"\"
    grid = state["grid"]
    start = tuple(state["me"])
    coins = {(r, c) for r, c, _ in state["coins"]}
    first_step = {start: None}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) in coins and (r, c) != start:
            return first_step[(r, c)]
        for k, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
                if (nr, nc) not in first_step:
                    first_step[(nr, nc)] = k if (r, c) == start else first_step[(r, c)]
                    queue.append((nr, nc))
    return 4
"""


@dataclass(frozen=True)
class GridInstance:
    height: int
    width: int
    walls: tuple[bool, ...]
    """Row-major, ``height * width`` cells."""
    coins: tuple[tuple[int, int], ...]
    """``(cell, value)`` pairs."""
    turns: int

    @property
    def starts(self) -> tuple[int, int]:
        return 0, self.height * self.width - 1


@functools.lru_cache(maxsize=256)
def _geometry(inst: GridInstance) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """All-pairs shortest path lengths (BFS) and the five move targets of every cell."""
    h, w = inst.height, inst.width
    n = h * w
    moves = np.empty((n, len(_MOVES)), dtype=np.int64)
    for cell in range(n):
        r, c = divmod(cell, w)
        for k, (dr, dc) in enumerate(_MOVES):
            nr, nc = r + dr, c + dc
            inside = 0 <= nr < h and 0 <= nc < w
            moves[cell, k] = nr * w + nc if inside and not inst.walls[nr * w + nc] else cell
    far = float(n)
    dist = np.full((n, n), far)
    for src in range(n):
        if inst.walls[src]:
            continue
        dist[src, src] = 0.0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt in moves[cur, :4]:
                if dist[src, nxt] == far:
                    dist[src, nxt] = dist[src, cur] + 1.0
                    queue.append(int(nxt))
    return dist, moves


class GridBots:
    """Two-player coin race on random point-symmetric maps."""

    name = "gridbots"

    def __init__(
        self, size: int = 11, coin_pairs: int = 12, wall_density: float = 0.2, turns: int = 80
    ) -> None:
        if size < 3 or coin_pairs < 1 or not 0.0 <= wall_density < 0.6 or turns < 1:
            raise ValueError("invalid gridbots options")
        self.size = size
        self.coin_pairs = coin_pairs
        self.wall_density = wall_density
        self.turns = turns

    def param_space(self) -> ParamSpace:
        return SPACE

    def baselines(self) -> list[ParamStrategy]:
        """Simple single-idea bots."""
        presets = {
            "nearest-coin": (1.0, 0.0, 0.0, 0.0, 0.0),
            "best-value": (0.0, 1.0, 0.0, 0.0, 0.0),
            "racer": (0.3, 0.0, 1.0, 0.0, 0.0),
            "cluster-seeker": (0.0, 0.0, 0.0, 1.0, 0.0),
        }
        return [ParamStrategy(name, values, SPACE) for name, values in presets.items()]

    def code_spec(self) -> CodeSpec:
        return CodeSpec("choose_move", CODE_DESCRIPTION, CODE_EXAMPLE, PARAM_GUIDE)

    def generate_instance(self, seed: int) -> GridInstance:
        rng = np.random.default_rng(seed)
        h = w = self.size
        n = h * w
        a, b = 0, n - 1
        protected = {a, b, 1, w, n - 2, n - 1 - w}
        while True:
            walls = [False] * n
            for cell in range(n // 2):
                if cell not in protected and rng.random() < self.wall_density:
                    walls[cell] = walls[n - 1 - cell] = True
            reach = self._reachable(walls, h, w, a)
            if b in reach:
                break
        walls = [cell not in reach for cell in range(n)]

        free_half = [c for c in range(n // 2) if not walls[c] and c not in protected]
        k = min(self.coin_pairs, len(free_half))
        coins: list[tuple[int, int]] = []
        for cell in sorted(int(x) for x in rng.choice(free_half, size=k, replace=False)):
            value = 3 if rng.random() < 0.2 else 1
            coins += [(cell, value), (n - 1 - cell, value)]
        return GridInstance(h, w, tuple(walls), tuple(sorted(coins)), self.turns)

    @staticmethod
    def _reachable(walls: list[bool], h: int, w: int, start: int) -> set[int]:
        seen = {start}
        queue = deque([start])
        while queue:
            r, c = divmod(queue.popleft(), w)
            for dr, dc in _MOVES[:4]:
                nr, nc = r + dr, c + dc
                nxt = nr * w + nc
                if 0 <= nr < h and 0 <= nc < w and not walls[nxt] and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def play(self, first: Strategy, second: Strategy, instance: GridInstance) -> float:
        dist, moves = _geometry(instance)
        scale = float(instance.height + instance.width)
        w = instance.width
        rows = [
            "".join("#" if instance.walls[r * w + c] else "." for c in range(w))
            for r in range(instance.height)
        ]
        pos = list(instance.starts)
        coins = dict(instance.coins)
        points = [0, 0]

        with contextlib.ExitStack() as stack:
            deciders: list[Callable[[int, int, int, int], int]] = []
            for i, strategy in enumerate((first, second)):
                if isinstance(strategy, CodeStrategy):
                    try:
                        fn = stack.enter_context(
                            SandboxedFunction(strategy.source, strategy.entrypoint, strategy.limits)
                        )
                    except SandboxError:
                        return self._forfeit(i)
                    deciders.append(
                        self._code_decider(fn, i, rows, w, moves, lambda: coins, points)
                    )
                else:
                    weights = np.asarray(strategy.values)
                    deciders.append(self._param_decider(weights, dist, moves, scale, lambda: coins))

            for t in range(instance.turns):
                if not coins:
                    break
                targets = []
                for i in (0, 1):
                    try:
                        targets.append(deciders[i](pos[i], pos[1 - i], t, instance.turns - t))
                    except (SandboxError, ValueError):
                        return self._forfeit(i)
                if targets[0] != targets[1]:
                    pos = targets
                for i in (0, 1):
                    points[i] += coins.pop(pos[i], 0)
        if points[0] == points[1]:
            return 0.5
        return 1.0 if points[0] > points[1] else 0.0

    @staticmethod
    def _forfeit(player: int) -> float:
        """Result for the first player when ``player`` broke the rules."""
        return 0.0 if player == 0 else 1.0

    def _param_decider(
        self,
        weights: npt.NDArray[np.float64],
        dist: npt.NDArray[np.float64],
        moves: npt.NDArray[np.int64],
        scale: float,
        coins: Callable[[], dict[int, int]],
    ) -> Callable[[int, int, int, int], int]:
        def decide(me: int, opp: int, turn: int, turns_left: int) -> int:
            board = coins()
            cells = np.fromiter(board, dtype=np.int64)
            values = np.fromiter(board.values(), dtype=float)
            return self._choose(weights, me, opp, cells, values, dist, moves, scale)

        return decide

    @staticmethod
    def _code_decider(
        fn: SandboxedFunction,
        player: int,
        rows: list[str],
        width: int,
        moves: npt.NDArray[np.int64],
        coins: Callable[[], dict[int, int]],
        points: list[int],
    ) -> Callable[[int, int, int, int], int]:
        def decide(me: int, opp: int, turn: int, turns_left: int) -> int:
            board = coins()
            state = {
                "grid": rows,
                "me": list(divmod(me, width)),
                "opponent": list(divmod(opp, width)),
                "coins": [[*divmod(c, width), v] for c, v in sorted(board.items())],
                "turn": turn,
                "turns_left": turns_left,
                "my_points": points[player],
                "opponent_points": points[1 - player],
            }
            move = fn(state)
            if not isinstance(move, int) or isinstance(move, bool) or not 0 <= move < 5:
                raise ValueError(f"invalid move {move!r}")
            return int(moves[me, move])

        return decide

    @staticmethod
    def _choose(
        weights: npt.NDArray[np.float64],
        me: int,
        opp: int,
        cells: npt.NDArray[np.int64],
        values: npt.NDArray[np.float64],
        dist: npt.NDArray[np.float64],
        moves: npt.NDArray[np.int64],
        scale: float,
    ) -> int:
        options = moves[me]
        my_d = dist[options][:, cells]
        opp_d = dist[opp, cells]
        total = values.sum()
        features = np.stack(
            [
                -my_d.min(axis=1) / scale,
                (values / (1.0 + my_d)).max(axis=1) / 3.0,
                ((values * (my_d < opp_d)).sum(axis=1) - (values * (my_d > opp_d)).sum(axis=1))
                / total,
                (values * np.exp(-my_d / 3.0)).sum(axis=1) / total,
                dist[options, opp] / scale,
            ],
            axis=1,
        )
        return int(options[int(np.argmax(features @ weights))])
