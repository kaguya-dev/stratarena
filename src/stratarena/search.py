"""The full search: arena, hybridization and elitism over generations, then validation."""

from dataclasses import dataclass

from stratarena.agents import AgentContext, AgentEvent, Orchestrator
from stratarena.arena import Arena, ArenaResult
from stratarena.config import RunConfig
from stratarena.evaluation import Evaluator
from stratarena.hybridizer import ParamHybridizer
from stratarena.llm import LLM, CachedLLM, Exchange, make_llm
from stratarena.log import get_logger
from stratarena.problem import AnyProblem, GameProblem, HasBaselines
from stratarena.seeding import Stream, derive_seed, make_rng
from stratarena.strategy import ParamStrategy, Strategy
from stratarena.validation import ValidationResult, validate

log = get_logger(__name__)


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    populations: tuple[tuple[Strategy, ...], ...]
    generations: tuple[ArenaResult, ...]
    baselines: tuple[ParamStrategy, ...]
    validation: ValidationResult
    tracking: tuple[dict[str, float], ...]
    """Per generation, each finalist's mean on the fixed tracking set."""
    baseline_tracking: dict[str, float]
    mode: str
    """``"score"`` for single-player problems, ``"game"`` for head-to-head games."""
    agent_events: tuple[AgentEvent, ...] = ()
    llm: str | None = None
    exchanges: tuple[Exchange, ...] = ()
    """Every LLM request and answer of the run (also cached on disk for replay)."""

    @property
    def champion(self) -> Strategy:
        return self.generations[-1].finalists[0]


def random_strategies(
    problem: AnyProblem, seed: int, generation: int, n: int
) -> list[ParamStrategy]:
    rng = make_rng(seed, Stream.INIT, generation)
    space = problem.param_space()
    return [ParamStrategy.random(f"g{generation}-r{i:02d}", space, rng) for i in range(n)]


def run_search(problem: AnyProblem, config: RunConfig, llm: LLM | None = None) -> RunResult:
    """Run every generation and validate the last finalists on held-out instances.

    Generation 0 is random (plus the problem's baselines when ``baselines_compete``).
    Each later generation keeps the previous finalists (elitism), adds their offspring and
    a few random immigrants to preserve diversity.

    Arena instances change every round, so training means of different generations are not
    comparable. To show progress, each generation's finalists are also scored on a fixed
    *tracking* set. It is reported only and never influences selection; the held-out set
    stays untouched until the end. In a game, tracking means the mean points per game
    against the fixed pool of baselines (or against the other finalists if the problem has
    no baselines).

    With ``config.agents.enabled``, an orchestrator also proposes candidates between
    generations (LLM-written code and tuned parameters); they replace part of the
    offspring and must survive the next arena like everyone else. ``llm`` overrides the
    configured model (useful for tests).
    """
    baselines = problem.baselines() if isinstance(problem, HasBaselines) else []
    is_game = isinstance(problem, GameProblem)
    arena = Arena(problem, config.arena, config.seed)
    hybridizer = ParamHybridizer(config.hybrid, config.seed)
    if llm is None and config.agents.enabled:
        llm = make_llm(config.agents.llm)
    orchestrator = Orchestrator(config.agents, llm) if config.agents.enabled else None
    agent_events: list[AgentEvent] = []
    feedback: list[str] = []

    competing = baselines if config.baselines_compete else []
    population: list[Strategy] = [
        *random_strategies(problem, config.seed, 0, config.population - len(competing)),
        *competing,
    ]
    populations: list[tuple[Strategy, ...]] = []
    results: list[ArenaResult] = []
    tracking: list[dict[str, float]] = []
    tracking_seeds = [
        derive_seed(config.seed, Stream.TRACKING, i) for i in range(config.tracking_instances)
    ]

    def tracked(strategies: list[Strategy]) -> dict[str, float]:
        opponents = baselines if is_game and baselines else None
        scores = evaluator.scores(problem, strategies, tracking_seeds, opponents)
        return {sid: sum(v) / len(v) for sid, v in scores.items()}

    with Evaluator(config.arena.workers) as evaluator:
        baseline_tracking = tracked(list(baselines)) if baselines else {}
        for g in range(config.generations):
            populations.append(tuple(population))
            result = arena.run(population, generation=g, evaluator=evaluator)
            results.append(result)
            tracking.append(tracked(list(result.finalists)))
            best = result.standings[0]
            log.info(
                "search.generation",
                generation=g,
                champion=best.strategy_id,
                mean=round(best.mean, 6),
                ci=(round(best.ci_low, 6), round(best.ci_high, 6)),
                tracking=round(tracking[-1][best.strategy_id], 6),
            )
            if g == config.generations - 1:
                break
            finalists = list(result.finalists)
            n_imm = config.hybrid.immigrants
            room = config.population - len(finalists) - n_imm
            proposed: list[Strategy] = []
            if orchestrator is not None:
                ctx = AgentContext(
                    problem=problem,
                    generation=g,
                    finalists=result.finalists,
                    standings=result.standings,
                    baselines=tuple(baselines),
                    seed=config.seed,
                    feedback=tuple(feedback),
                )
                proposal = orchestrator.propose(ctx)
                agent_events += proposal.events
                feedback += proposal.notes
                proposed = list(proposal.strategies)[:room]
            param_parents = [f for f in finalists if isinstance(f, ParamStrategy)]
            n_children = room - len(proposed)
            if not param_parents:
                n_imm, n_children = n_imm + n_children, 0
            population = [
                *finalists,
                *proposed,
                *hybridizer.offspring(param_parents, n_children, g),
                *random_strategies(problem, config.seed, g + 1, n_imm),
            ]

        last = results[-1]
        train_means = {s.strategy_id: s.mean for s in last.standings}
        validation = validate(
            problem,
            last.finalists,
            train_means,
            baselines,
            config.seed,
            config.holdout_instances,
            config.arena.bootstrap_samples,
            evaluator,
        )

    champ = validation.champion
    log.info(
        "search.validation",
        champion=validation.champion_id,
        holdout_mean=round(champ.holdout.mean, 6),
        gap=round(validation.generalization_gap, 6),
        overfit_suspected=validation.overfit_suspected,
    )
    return RunResult(
        config=config,
        populations=tuple(populations),
        generations=tuple(results),
        baselines=tuple(baselines),
        validation=validation,
        tracking=tuple(tracking),
        baseline_tracking=baseline_tracking,
        mode="game" if is_game else "score",
        agent_events=tuple(agent_events),
        llm=llm.name if llm is not None else None,
        exchanges=tuple(llm.exchanges) if isinstance(llm, CachedLLM) else (),
    )
