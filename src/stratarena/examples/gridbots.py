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

import functools
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from stratarena.strategy import ParamSpace, ParamSpec, ParamStrategy

SPACE: ParamSpace = (
    ParamSpec("near", -1.0, 1.0),
    ParamSpec("value", -1.0, 1.0),
    ParamSpec("race", -1.0, 1.0),
    ParamSpec("density", -1.0, 1.0),
    ParamSpec("spacing", -1.0, 1.0),
)

# Move order also breaks ties: N, S, W, E, stay.
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


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

    def play(self, first: ParamStrategy, second: ParamStrategy, instance: GridInstance) -> float:
        dist, moves = _geometry(instance)
        scale = float(instance.height + instance.width)
        weights = (np.asarray(first.values), np.asarray(second.values))
        pos = list(instance.starts)
        coins = dict(instance.coins)
        points = [0, 0]
        for _ in range(instance.turns):
            if not coins:
                break
            cells = np.fromiter(coins, dtype=np.int64)
            values = np.fromiter(coins.values(), dtype=float)
            targets = [
                self._choose(weights[i], pos[i], pos[1 - i], cells, values, dist, moves, scale)
                for i in (0, 1)
            ]
            if targets[0] != targets[1]:
                pos = targets
            for i in (0, 1):
                points[i] += coins.pop(pos[i], 0)
        if points[0] == points[1]:
            return 0.5
        return 1.0 if points[0] > points[1] else 0.0

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
