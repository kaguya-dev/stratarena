"""Shared types for the agents that propose candidates between generations."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from stratarena.arena import Standing
from stratarena.evaluation import Evaluator
from stratarena.problem import AnyProblem, GameProblem
from stratarena.seeding import Stream, derive_seed
from stratarena.strategy import CodeStrategy, ParamStrategy, Strategy


@dataclass(frozen=True)
class AgentContext:
    """What every agent sees after a generation's arena."""

    problem: AnyProblem
    generation: int
    """The generation that just finished; proposals join generation + 1."""
    finalists: tuple[Strategy, ...]
    standings: tuple[Standing, ...]
    baselines: tuple[ParamStrategy, ...]
    seed: int
    feedback: tuple[str, ...] = ()
    """Reviewer notes from earlier generations, oldest first."""

    @property
    def is_game(self) -> bool:
        return isinstance(self.problem, GameProblem)


@dataclass(frozen=True)
class AgentEvent:
    agent: str
    generation: int
    action: str
    """``accepted``, ``rejected``, ``reviewed``, ``proposed`` or ``failed``."""
    strategy_id: str | None
    detail: str


@dataclass(frozen=True)
class Proposal:
    strategies: tuple[Strategy, ...] = ()
    events: tuple[AgentEvent, ...] = ()
    notes: tuple[str, ...] = ()

    @staticmethod
    def merge(parts: Sequence["Proposal"]) -> "Proposal":
        return Proposal(
            strategies=tuple(s for p in parts for s in p.strategies),
            events=tuple(e for p in parts for e in p.events),
            notes=tuple(n for p in parts for n in p.notes),
        )


@dataclass(frozen=True)
class SmokeResult:
    scores: tuple[float, ...]
    error: str | None = None
    opponents: tuple[str, ...] = field(default=())

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


def smoke_test(ctx: AgentContext, strategy: CodeStrategy, n_instances: int) -> SmokeResult:
    """Run a new code strategy on a few dedicated instances before it may compete.

    Score problems reject any instance that errors. Games cannot tell a crash from a loss
    (a crashing bot forfeits), so a game candidate must take at least one point against
    the baselines (or the finalists if there are none).
    """
    seeds = [derive_seed(ctx.seed, Stream.SMOKE, ctx.generation, i) for i in range(n_instances)]
    if not isinstance(ctx.problem, GameProblem):
        scores: list[float] = []
        for seed in seeds:
            result = ctx.problem.evaluate(strategy, ctx.problem.generate_instance(seed))
            if result.error is not None:
                return SmokeResult(tuple(scores), error=result.error)
            scores.append(result.score)
        return SmokeResult(tuple(scores))

    opponents: list[Strategy] = list(ctx.baselines) or list(ctx.finalists)
    per_seed = Evaluator().scores(ctx.problem, [strategy], seeds, opponents=opponents)
    game_scores = tuple(per_seed[strategy.id])
    error = None if sum(game_scores) > 0 else "took no point in any smoke game (crash or forfeit)"
    return SmokeResult(game_scores, error, tuple(o.id for o in opponents))


def describe_strategy(strategy: Strategy, max_chars: int = 3000) -> str:
    """Short text description of a strategy for prompts."""
    if isinstance(strategy, ParamStrategy):
        params = ", ".join(f"{k}={v:.3f}" for k, v in strategy.as_dict().items())
        return f"parametric heuristic ({params})"
    source = strategy.source
    if len(source) > max_chars:
        source = source[:max_chars] + "\n# ... (truncated)"
    return f"code strategy:\n```python\n{source}\n```"
