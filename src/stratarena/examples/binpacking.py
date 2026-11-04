"""One-dimensional bin packing with a parametric placement heuristic.

The strategy has three parameters that span the classic heuristics:

- ``sort_fraction``: share of the items, largest first, that are placed before the rest
  (which keep their arrival order). 0 is a pure online heuristic, 1 is "decreasing".
- ``fit_weight``: preference for the bin left with the least (positive, best fit) or the
  most (negative, worst fit) free space after placing the item.
- ``first_weight``: preference for older bins (first fit).

Score is ``lower_bound / bins_used``, where ``lower_bound = ceil(sum(items) / capacity)``,
so 1.0 is provably optimal and scores are comparable across instances.
"""

import math
from dataclasses import dataclass

import numpy as np

from stratarena.problem import EvalResult
from stratarena.strategy import ParamSpace, ParamSpec, ParamStrategy

SPACE: ParamSpace = (
    ParamSpec("sort_fraction", 0.0, 1.0),
    ParamSpec("fit_weight", -1.0, 1.0),
    ParamSpec("first_weight", 0.0, 1.0),
)


@dataclass(frozen=True)
class BinPackingInstance:
    capacity: int
    items: tuple[int, ...]

    @property
    def lower_bound(self) -> int:
        return math.ceil(sum(self.items) / self.capacity)


class BinPacking:
    """Bin packing instances drawn from three item-size profiles (small, large, bimodal)."""

    name = "binpacking"

    def __init__(self, n_items: int = 120, capacity: int = 100) -> None:
        if n_items < 1 or capacity < 2:
            raise ValueError("n_items must be >= 1 and capacity >= 2")
        self.n_items = n_items
        self.capacity = capacity

    def param_space(self) -> ParamSpace:
        return SPACE

    def generate_instance(self, seed: int) -> BinPackingInstance:
        rng = np.random.default_rng(seed)
        c = self.capacity
        profile = int(rng.integers(3))
        if profile == 0:
            sizes = rng.uniform(0.05 * c, 0.6 * c, self.n_items)
        elif profile == 1:
            sizes = rng.uniform(0.2 * c, 0.7 * c, self.n_items)
        else:
            small = rng.uniform(0.05 * c, 0.25 * c, self.n_items)
            large = rng.uniform(0.5 * c, 0.8 * c, self.n_items)
            sizes = np.where(rng.random(self.n_items) < 0.5, small, large)
        items = np.clip(np.rint(sizes), 1, c).astype(int)
        return BinPackingInstance(capacity=c, items=tuple(int(x) for x in items))

    def pack(self, strategy: ParamStrategy, instance: BinPackingInstance) -> list[list[int]]:
        """Place every item and return the bins (each a list of item sizes)."""
        items = instance.items
        n = len(items)
        k = round(strategy["sort_fraction"] * n)
        by_size = sorted(range(n), key=lambda i: (-items[i], i))
        first = by_size[:k]
        chosen = set(first)
        order = first + [i for i in range(n) if i not in chosen]

        fit_w = strategy["fit_weight"]
        first_w = strategy["first_weight"]
        cap = instance.capacity
        residual = np.empty(n, dtype=float)
        bins: list[list[int]] = []
        for i in order:
            size = items[i]
            m = len(bins)
            feasible = np.flatnonzero(residual[:m] >= size)
            if feasible.size == 0:
                residual[m] = cap - size
                bins.append([size])
                continue
            after = residual[feasible] - size
            priority = -fit_w * after / cap - first_w * feasible / m
            b = int(feasible[int(np.argmax(priority))])
            residual[b] -= size
            bins[b].append(size)
        return bins

    def evaluate(self, strategy: ParamStrategy, instance: BinPackingInstance) -> EvalResult:
        bins = self.pack(strategy, instance)
        lb = instance.lower_bound
        return EvalResult(
            score=lb / len(bins),
            metrics={"bins": float(len(bins)), "lower_bound": float(lb)},
        )
