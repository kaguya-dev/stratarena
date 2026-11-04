"""Parameter tuner: local search around the best parametric finalist. Needs no LLM."""

import numpy as np

from stratarena.agents.base import AgentContext, AgentEvent, Proposal
from stratarena.evaluation import Evaluator
from stratarena.seeding import Stream, derive_seed, make_rng
from stratarena.strategy import ParamStrategy, Strategy, clip_to_space


class ParamTunerAgent:
    """A (1+λ) evolution strategy with a 1/5-style step-size rule.

    It scores candidates on its own instances (a separate seed stream, never the arena's
    or the held-out set) and proposes the best points that beat the starting finalist.
    For games, fitness is the mean points against the baselines (or the finalists).
    Fully deterministic for a given seed and generation.
    """

    name = "tuner"

    def __init__(self, iterations: int = 10, instances: int = 6, offspring: int = 4) -> None:
        self.iterations = iterations
        self.instances = instances
        self.offspring = offspring

    def propose(self, ctx: AgentContext, n: int) -> Proposal:
        params = [f for f in ctx.finalists if isinstance(f, ParamStrategy)]
        if n == 0 or not params:
            return Proposal()
        base = params[0]
        g = ctx.generation
        seeds = [derive_seed(ctx.seed, Stream.TUNING, g, 1, i) for i in range(self.instances)]
        rng = make_rng(ctx.seed, Stream.TUNING, g, 0)
        opponents: list[Strategy] | None = None
        if ctx.is_game:
            opponents = list(ctx.baselines) or list(ctx.finalists)
        evaluator = Evaluator()

        def fitness(cands: list[ParamStrategy]) -> list[float]:
            scores = evaluator.scores(ctx.problem, cands, seeds, opponents)
            return [float(np.mean(scores[c.id])) for c in cands]

        space = base.space
        span = np.array([p.high - p.low for p in space])
        current, (current_fit,) = base, fitness([base])
        base_fit = current_fit
        sigma = 0.15
        found: list[tuple[float, ParamStrategy]] = []
        for it in range(self.iterations):
            cands = [
                ParamStrategy(
                    f"tune-{it}-{j}",
                    clip_to_space(
                        space, (np.asarray(current.values) + rng.normal(0, sigma * span)).tolist()
                    ),
                    space,
                )
                for j in range(self.offspring)
            ]
            fits = fitness(cands)
            best = int(np.argmax(fits))
            if fits[best] > current_fit:
                current, current_fit = cands[best], fits[best]
                found.append((current_fit, current))
                sigma = min(0.5, sigma * 1.5)
            else:
                sigma = max(0.01, sigma * 0.8)

        found.sort(key=lambda x: -x[0])
        strategies = [
            ParamStrategy(f"g{g + 1}-t{i:02d}", s.values, space, parents=(base.id,))
            for i, (_, s) in enumerate(found[:n])
        ]
        detail = (
            f"from `{base.id}` ({base_fit:.4f}) to {found[0][0]:.4f} on {self.instances} "
            "tuning instances"
            if found
            else f"no improvement over `{base.id}` ({base_fit:.4f})"
        )
        events = tuple(AgentEvent(self.name, g, "proposed", s.id, detail) for s in strategies) or (
            AgentEvent(self.name, g, "proposed", None, detail),
        )
        return Proposal(tuple(strategies), events)
