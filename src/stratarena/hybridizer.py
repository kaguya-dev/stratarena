"""Hybridization: offspring of the finalists.

- ``ParamHybridizer``: blend crossover plus Gaussian mutation of parameter vectors.
- ``CodeFusionHybridizer``: an LLM merges two finalists (code or parametric) into a new
  code strategy, admitted only after it loads in the sandbox and passes the smoke test.

Offspring rejoin the arena next to their parents (the finalists are kept), so a child
survives only if it actually beats them.
"""

from collections.abc import Sequence

from stratarena.agents.base import AgentContext, Proposal, describe_strategy
from stratarena.agents.generator import write_code
from stratarena.config import HybridConfig
from stratarena.llm import LLM
from stratarena.problem import HasCodeSpec
from stratarena.sandbox import SandboxLimits
from stratarena.seeding import Stream, make_rng
from stratarena.strategy import ParamStrategy, Strategy, clip_to_space


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


def fusion_pairs(n_finalists: int, k: int) -> list[tuple[int, int]]:
    """The first ``k`` pairs of finalist ranks, best pairs first: (0,1), (0,2), (1,2), ..."""
    pairs = [(i, j) for j in range(1, n_finalists) for i in range(j)]
    return pairs[:k]


class CodeFusionHybridizer:
    """LLM-assisted crossover for code: two parents in, one tested child out."""

    name = "fusion"

    def __init__(
        self, llm: LLM, limits: SandboxLimits, max_attempts: int = 2, smoke_instances: int = 2
    ) -> None:
        self.llm = llm
        self.limits = limits
        self.max_attempts = max_attempts
        self.smoke_instances = smoke_instances

    def prompt(self, ctx: AgentContext, a: Strategy, b: Strategy) -> str:
        assert isinstance(ctx.problem, HasCodeSpec)
        spec = ctx.problem.code_spec()
        means = {s.strategy_id: s.mean for s in ctx.standings}
        guide = (
            f"## How to read parametric heuristics\n\n{spec.param_guide}\n"
            if spec.param_guide
            else ""
        )
        parents = "\n\n".join(
            f"### Parent `{p.id}` (mean score {means.get(p.id, float('nan')):.4f})\n\n"
            f"{describe_strategy(p)}"
            for p in (a, b)
        )
        return (
            f"# Problem: {ctx.problem.name}\n\n{spec.description}\n{guide}"
            f"## Two strong strategies\n\n{parents}\n\n"
            "## Your task\n\nWrite one new `"
            f"{spec.entrypoint}` that combines the strengths of both parents: keep the idea "
            "that makes each of them score well and drop what does not help. The result "
            "must be a single self-contained implementation."
        )

    def fuse(self, ctx: AgentContext, index: int, a: Strategy, b: Strategy) -> Proposal:
        if not isinstance(ctx.problem, HasCodeSpec):
            return Proposal()
        return write_code(
            llm=self.llm,
            agent=self.name,
            ctx=ctx,
            spec=ctx.problem.code_spec(),
            sid=f"g{ctx.generation + 1}-fx{index:02d}",
            parents=(a.id, b.id),
            prompt=self.prompt(ctx, a, b),
            limits=self.limits,
            max_attempts=self.max_attempts,
            smoke_instances=self.smoke_instances,
        )
