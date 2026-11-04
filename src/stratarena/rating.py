"""Head-to-head ratings: Bradley-Terry strengths reported on the Elo scale.

Classic Elo updates depend on the order in which games are processed. Bradley-Terry
fits all games at once, so the rating does not depend on scheduling or on the number of
worker processes; it is shown on the familiar Elo scale (400 points = 10:1 odds,
average 1500).
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

ELO_MEAN = 1500.0


@dataclass
class PairStats:
    """Points and games between pairs of strategies. Draws count half a point each."""

    points: dict[tuple[str, str], float] = field(default_factory=dict)
    games: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(self, a: str, b: str, points_a: float, games: int) -> None:
        """Record ``games`` between ``a`` and ``b`` in which ``a`` scored ``points_a``."""
        for x, y, pts in ((a, b, points_a), (b, a, games - points_a)):
            self.points[(x, y)] = self.points.get((x, y), 0.0) + pts
            self.games[(x, y)] = self.games.get((x, y), 0) + games

    def merge(self, other: "PairStats") -> None:
        for key, pts in other.points.items():
            self.points[key] = self.points.get(key, 0.0) + pts
        for key, n in other.games.items():
            self.games[key] = self.games.get(key, 0) + n

    def ratings(self, ids: Iterable[str], iterations: int = 500) -> dict[str, float]:
        """Elo-scale Bradley-Terry ratings of ``ids``, using only games among them."""
        return bradley_terry(sorted(set(ids)), self, iterations)


def bradley_terry(ids: list[str], stats: PairStats, iterations: int = 500) -> dict[str, float]:
    """Fit strengths with the MM algorithm (Hunter, 2004).

    Every pair that played gets one extra virtual drawn game, a light prior that keeps
    ratings finite when a strategy won or lost every game.
    """
    n = len(ids)
    games = [[0.0] * n for _ in range(n)]
    wins = [0.0] * n
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            g = stats.games.get((a, b), 0)
            if i == j or g == 0:
                continue
            games[i][j] = g + 1.0
            wins[i] += stats.points[(a, b)] + 0.5

    strength = [1.0] * n
    for _ in range(iterations):
        new = []
        for i in range(n):
            denom = sum(games[i][j] / (strength[i] + strength[j]) for j in range(n) if games[i][j])
            new.append(wins[i] / denom if denom else strength[i])
        log_mean = sum(math.log(s) for s in new) / n if n else 0.0
        new = [s / math.exp(log_mean) for s in new]
        converged = max((abs(x - y) for x, y in zip(new, strength, strict=True)), default=0.0)
        strength = new
        if converged < 1e-10:
            break
    return {a: ELO_MEAN + 400.0 * math.log10(s) for a, s in zip(ids, strength, strict=True)}
