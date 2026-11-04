"""The full search: arena, hybridization and elitism over generations, then validation."""

from dataclasses import dataclass
from typing import Any

from stratarena.arena import Arena, ArenaResult
from stratarena.config import RunConfig
from stratarena.evaluation import Evaluator
from stratarena.hybridizer import ParamHybridizer
from stratarena.log import get_logger
from stratarena.problem import HasBaselines, Problem
from stratarena.seeding import Stream, derive_seed, make_rng
from stratarena.strategy import ParamStrategy
from stratarena.validation import ValidationResult, validate

log = get_logger(__name__)


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    populations: tuple[tuple[ParamStrategy, ...], ...]
    generations: tuple[ArenaResult, ...]
    baselines: tuple[ParamStrategy, ...]
    validation: ValidationResult
    tracking: tuple[dict[str, float], ...]
    """Per generation, each finalist's mean on the fixed tracking set."""
    baseline_tracking: dict[str, float]

    @property
    def champion(self) -> ParamStrategy:
        return self.generations[-1].finalists[0]


def random_strategies(
    problem: Problem[Any], seed: int, generation: int, n: int
) -> list[ParamStrategy]:
    rng = make_rng(seed, Stream.INIT, generation)
    space = problem.param_space()
    return [ParamStrategy.random(f"g{generation}-r{i:02d}", space, rng) for i in range(n)]


def run_search(problem: Problem[Any], config: RunConfig) -> RunResult:
    """Run every generation and validate the last finalists on held-out instances.

    Generation 0 is random (plus the problem's baselines when ``baselines_compete``).
    Each later generation keeps the previous finalists (elitism), adds their offspring and
    a few random immigrants to preserve diversity.

    Arena instances change every round, so training means of different generations are not
    comparable. To show progress, each generation's finalists are also scored on a fixed
    *tracking* set. It is reported only and never influences selection; the held-out set
    stays untouched until the end.
    """
    baselines = problem.baselines() if isinstance(problem, HasBaselines) else []
    arena = Arena(problem, config.arena, config.seed)
    hybridizer = ParamHybridizer(config.hybrid, config.seed)

    competing = baselines if config.baselines_compete else []
    population = [
        *random_strategies(problem, config.seed, 0, config.population - len(competing)),
        *competing,
    ]
    populations: list[tuple[ParamStrategy, ...]] = []
    results: list[ArenaResult] = []
    tracking: list[dict[str, float]] = []
    tracking_seeds = [
        derive_seed(config.seed, Stream.TRACKING, i) for i in range(config.tracking_instances)
    ]

    def tracked(strategies: list[ParamStrategy]) -> dict[str, float]:
        scores = evaluator.scores(problem, strategies, tracking_seeds)
        return {sid: sum(v) / len(v) for sid, v in scores.items()}

    with Evaluator(config.arena.workers) as evaluator:
        baseline_tracking = tracked(baselines) if baselines else {}
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
            n_children = config.population - len(finalists) - n_imm
            population = [
                *finalists,
                *hybridizer.offspring(finalists, n_children, g),
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
    )
