"""One-dimensional bin packing with a parametric placement heuristic.

Each item goes to the feasible bin with the highest priority, a weighted sum of features of
the placement. The first three parameters span the classic heuristics (first/best/worst
fit, with or without sorting); the next three look at the items still to be placed, and
the last one lets the strategy open a new bin on purpose:

- ``sort_fraction``: share of the items, largest first, that are placed before the rest
  (which keep their arrival order). 0 is a pure online heuristic, 1 is "decreasing".
- ``fit_weight``: preference for the bin left with the least (positive, best fit) or the
  most (negative, worst fit) free space after placing the item.
- ``first_weight``: preference for older bins (first fit).
- ``exact_bonus``: bonus when the item fills the bin exactly.
- ``waste_penalty``: penalty when the space left is smaller than every remaining item,
  i.e. guaranteed to stay empty.
- ``fillable_bonus``: bonus when the space left equals the size of some remaining item,
  so a later item can close the bin exactly.
- ``open_penalty``: opening a new bin is always a candidate, scored with the same features
  minus this penalty. At the maximum (5) a new bin is opened only when nothing fits, as in
  the classic heuristics; lower values let the strategy refuse a bad fit.

The classic heuristics are the points with the three lookahead parameters at zero and
``open_penalty`` at its maximum.

Score is ``reference / bins_used``. The reference is the known optimum for the "triplets"
profile, whose instances are built from bins filled exactly by three items (the classic
class on which first/best fit decreasing are known to be suboptimal), and otherwise the
lower bound ``ceil(sum(items) / capacity)``, which is weak for large items: on those
profiles a score below 1.0 does not mean a better packing exists.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stratarena.problem import CodeSpec, EvalResult
from stratarena.sandbox import SandboxedFunction, SandboxError
from stratarena.strategy import CodeStrategy, ParamSpace, ParamSpec, ParamStrategy, Strategy

SPACE: ParamSpace = (
    ParamSpec("sort_fraction", 0.0, 1.0),
    ParamSpec("fit_weight", -1.0, 1.0),
    ParamSpec("first_weight", 0.0, 1.0),
    ParamSpec("exact_bonus", 0.0, 1.0),
    ParamSpec("waste_penalty", 0.0, 1.0),
    ParamSpec("fillable_bonus", 0.0, 1.0),
    ParamSpec("open_penalty", 0.0, 5.0),
)


CODE_DESCRIPTION = """\
Write a function `pack(items: list[int], capacity: int) -> list[int]`.

- `items` are the item sizes, each between 1 and `capacity`, in arrival order.
- Return a list with one bin number per item (same length and order as `items`); bin
  numbers are any non-negative integers, items with the same number share a bin.
- The total size in every bin must not exceed `capacity`.
- Goal: use as few bins as possible. Score = reference / bins used (1.0 is optimal).
- Standard library only (no numpy), at most 2 seconds per call, no files or network.
- An invalid answer (wrong length, overfull bin, exception, timeout) scores 0.
"""

CODE_EXAMPLE = """\
def pack(items, capacity):
    \"\"\"First fit decreasing.\"\"\"
    order = sorted(range(len(items)), key=lambda i: -items[i])
    free = []
    assignment = [0] * len(items)
    for i in order:
        for b, room in enumerate(free):
            if room >= items[i]:
                free[b] -= items[i]
                assignment[i] = b
                break
        else:
            free.append(capacity - items[i])
            assignment[i] = len(free) - 1
    return assignment
"""


PARAM_GUIDE = """\
A parametric heuristic first places the largest `round(sort_fraction * n)` items in
decreasing size, then the rest in arrival order. For each item it scores every bin the
item fits in, plus the option of opening a new bin, and takes the highest score:

    score = -fit_weight * space_left_after / capacity
            - first_weight * (bin_index / number_of_open_bins)    # new bin counts as 1.0
            + exact_bonus     if space_left_after == 0
            - waste_penalty   if 0 < space_left_after < smallest item still to place
            + fillable_bonus  if space_left_after equals the size of an item still to place
            - open_penalty    for the "open a new bin" option only

`open_penalty` near 5 means a new bin is opened only when nothing fits (classic first or
best fit). A low `open_penalty` means the heuristic deliberately refuses bad fits and
opens a new bin instead, which pays off when items must be grouped in exact triples.
"""


@dataclass(frozen=True)
class BinPackingInstance:
    capacity: int
    items: tuple[int, ...]
    optimum: int | None = None

    @property
    def lower_bound(self) -> int:
        return math.ceil(sum(self.items) / self.capacity)

    @property
    def reference(self) -> int:
        """Known optimum when available, else the lower bound."""
        return self.optimum if self.optimum is not None else self.lower_bound


class BinPacking:
    """Bin packing instances drawn from four item-size profiles.

    The profiles are small, large, triplets (known optimum) and bimodal items.
    """

    name = "binpacking"
    PROFILES = ("small", "large", "triplets", "bimodal")

    def __init__(
        self, n_items: int = 120, capacity: int = 100, profiles: Sequence[str] = PROFILES
    ) -> None:
        if n_items < 1 or capacity < 2:
            raise ValueError("n_items must be >= 1 and capacity >= 2")
        unknown = set(profiles) - set(self.PROFILES)
        if not profiles or unknown:
            raise ValueError(f"profiles must be a non-empty subset of {self.PROFILES}")
        self.n_items = n_items
        self.capacity = capacity
        self.profiles = tuple(profiles)

    def param_space(self) -> ParamSpace:
        return SPACE

    def baselines(self) -> list[ParamStrategy]:
        """The four classic heuristics, expressed as points of the parameter space."""
        presets = {
            "first-fit": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0),
            "best-fit": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0),
            "first-fit-decreasing": (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0),
            "best-fit-decreasing": (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0),
        }
        return [ParamStrategy(name, values, SPACE) for name, values in presets.items()]

    def code_spec(self) -> CodeSpec:
        return CodeSpec("pack", CODE_DESCRIPTION, CODE_EXAMPLE, PARAM_GUIDE)

    def generate_instance(self, seed: int) -> BinPackingInstance:
        rng = np.random.default_rng(seed)
        c = self.capacity
        profile = self.profiles[int(rng.integers(len(self.profiles)))]
        if profile == "small":
            sizes = rng.uniform(0.05 * c, 0.6 * c, self.n_items)
        elif profile == "large":
            sizes = rng.uniform(0.2 * c, 0.7 * c, self.n_items)
        elif profile == "triplets":
            return self._triplets(rng)
        else:
            small = rng.uniform(0.05 * c, 0.25 * c, self.n_items)
            large = rng.uniform(0.5 * c, 0.8 * c, self.n_items)
            sizes = np.where(rng.random(self.n_items) < 0.5, small, large)
        items = np.clip(np.rint(sizes), 1, c).astype(int)
        return BinPackingInstance(capacity=c, items=tuple(int(x) for x in items))

    def _triplets(self, rng: np.random.Generator) -> BinPackingInstance:
        """Bins split into three items that sum exactly to the capacity, then shuffled."""
        c = self.capacity
        n_bins = max(1, self.n_items // 3)
        items: list[int] = []
        for _ in range(n_bins):
            a = int(rng.integers(round(0.38 * c), c // 2))
            b = int(rng.integers(c // 4, c - a - c // 4 + 1))
            items += [a, b, c - a - b]
        order = rng.permutation(len(items))
        return BinPackingInstance(c, tuple(items[i] for i in order), optimum=n_bins)

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
        exact_b = strategy["exact_bonus"]
        waste_p = strategy["waste_penalty"]
        fill_b = strategy["fillable_bonus"]
        open_p = strategy["open_penalty"]
        cap = instance.capacity
        # remaining[s] = items of size s not placed yet; smallest = smallest such size.
        remaining = np.bincount(np.asarray(items, dtype=np.int64), minlength=cap + 1)
        smallest = 0
        residual = np.empty(n, dtype=np.int64)
        bins: list[list[int]] = []
        for i in order:
            size = items[i]
            remaining[size] -= 1
            while smallest <= cap and remaining[smallest] == 0:
                smallest += 1
            m = len(bins)
            feasible = np.flatnonzero(residual[:m] >= size)
            # Candidates: every feasible bin, then a new bin (index m, age 1.0). On ties
            # argmax keeps the first, so an existing bin wins over opening a new one.
            after = np.append(residual[feasible] - size, cap - size)
            age = np.append(feasible / max(m, 1), 1.0)
            gap = after > 0
            priority = (
                -fit_w * after / cap
                - first_w * age
                + exact_b * (after == 0)
                - waste_p * (gap & (after < smallest))
                + fill_b * (gap & (remaining[after] > 0))
            )
            priority[-1] -= open_p
            choice = int(np.argmax(priority))
            if choice == feasible.size:
                residual[m] = cap - size
                bins.append([size])
            else:
                b = int(feasible[choice])
                residual[b] -= size
                bins[b].append(size)
        return bins

    def pack_with_code(
        self, strategy: CodeStrategy, instance: BinPackingInstance
    ) -> list[list[int]]:
        """Run the strategy's ``pack`` in the sandbox and validate its assignment."""
        items = list(instance.items)
        with SandboxedFunction(strategy.source, strategy.entrypoint, strategy.limits) as fn:
            assignment = fn(items, instance.capacity)
        if not isinstance(assignment, list) or len(assignment) != len(items):
            raise ValueError("pack must return one bin number per item")
        bins: dict[int, list[int]] = {}
        for size, b in zip(items, assignment, strict=True):
            if not isinstance(b, int) or isinstance(b, bool) or b < 0:
                raise ValueError(f"invalid bin number {b!r}")
            bins.setdefault(b, []).append(size)
        for content in bins.values():
            if sum(content) > instance.capacity:
                raise ValueError("a bin exceeds the capacity")
        return [bins[b] for b in sorted(bins)]

    def evaluate(self, strategy: Strategy, instance: BinPackingInstance) -> EvalResult:
        if isinstance(strategy, CodeStrategy):
            try:
                bins = self.pack_with_code(strategy, instance)
            except (SandboxError, ValueError) as e:
                return EvalResult(score=0.0, metrics={"error": 1.0}, error=str(e))
        else:
            bins = self.pack(strategy, instance)
        return EvalResult(
            score=instance.reference / len(bins),
            metrics={
                "bins": float(len(bins)),
                "lower_bound": float(instance.lower_bound),
                "reference": float(instance.reference),
            },
        )
