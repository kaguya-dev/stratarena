"""Offspring of parametric finalists: blend crossover plus Gaussian mutation."""

from collections.abc import Sequence

from stratarena.config import HybridConfig
from stratarena.seeding import Stream, make_rng
from stratarena.strategy import ParamStrategy, clip_to_space


class ParamHybridizer:
    """Creates children from pairs of parents.

    Each gene is drawn from the BLX-alpha interval around the two parent values, then,
    with probability ``mutation_rate``, shifted by Gaussian noise whose scale is
    ``mutation_scale`` times the parameter's range. Child ``i`` of a generation always uses
    the same random stream, so offspring are reproducible.
    """

    def __init__(self, config: HybridConfig, seed: int) -> None:
        self.config = config
        self.seed = seed

    def offspring(
        self, parents: Sequence[ParamStrategy], n: int, generation: int
    ) -> list[ParamStrategy]:
        if n == 0:
            return []
        if not parents:
            raise ValueError("need at least one parent")
        cfg = self.config
        children = []
        for i in range(n):
            rng = make_rng(self.seed, Stream.VARIATION, generation, i)
            if len(parents) > 1:
                a, b = (int(x) for x in rng.choice(len(parents), size=2, replace=False))
            else:
                a = b = 0
            pa, pb = parents[a], parents[b]
            values = []
            for x, y, spec in zip(pa.values, pb.values, pa.space, strict=True):
                lo, hi = min(x, y), max(x, y)
                margin = cfg.crossover_alpha * (hi - lo)
                v = float(rng.uniform(lo - margin, hi + margin))
                if rng.random() < cfg.mutation_rate:
                    v += float(rng.normal(0.0, cfg.mutation_scale * (spec.high - spec.low)))
                values.append(v)
            children.append(
                ParamStrategy(
                    id=f"g{generation + 1}-c{i:02d}",
                    values=clip_to_space(pa.space, values),
                    space=pa.space,
                    parents=tuple(dict.fromkeys((pa.id, pb.id))),
                )
            )
        return children
